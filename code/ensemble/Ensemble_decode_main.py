"""
Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
All models in the ensemble must share a target vocabulary.
"""

import torch
import torch.nn as nn
import argparse
import yaml
import os

from inference import Ensemble_Translate_Transformer
from model import EmbedPatchAttention, make_model_onlySpec, make_model_withFormula
from utils import score_main, PositionalEncoding


class EnsembleDecoderOutput(object):
    """Wrapper around multiple decoder final hidden states."""

    def __init__(self, model_dec_outs):
        self.model_dec_outs = tuple(model_dec_outs)

    def squeeze(self, dim=None):
        """Delegate squeeze to avoid modifying
        :func:`onmt.translate.translator.Translator.translate_batch()`
        """
        return EnsembleDecoderOutput([x.squeeze(dim) for x in self.model_dec_outs])

    def __getitem__(self, index):
        return self.model_dec_outs[index]


class EnsembleEncoder(nn.Module):
    """Dummy Encoder that delegates to individual real Encoders."""

    def __init__(self, model_encoders, src_embeds):
        super(EnsembleEncoder, self).__init__()
        self.model_encoders = nn.ModuleList(model_encoders)
        self.src_embeds = nn.ModuleList(src_embeds)
        self.src_position = PositionalEncoding(d_model=512, dropout=0.1)

    def forward(self, formula, spec, src_mask):
        enc_out = [model_encoder(self.src_position(src_embed(formula, spec)),src_mask) for model_encoder,src_embed in zip(self.model_encoders, self.src_embeds)]
        return enc_out

class EnsembleDecoder(nn.Module):
    """Dummy Decoder that delegates to individual real Decoders."""

    def __init__(self, model_decoders, tgt_embeds):
        super(EnsembleDecoder, self).__init__()
        model_decoders = nn.ModuleList(model_decoders)
        self.model_decoders = model_decoders
        self.tgt_embeds = nn.ModuleList(tgt_embeds)


    def forward(self, enc_out, tgt, src_mask, tgt_mask):
        

        dec_outs = [
            model_decoder(tgt_embed(tgt), enc_out[i], src_mask, tgt_mask) for i, (model_decoder, tgt_embed) in enumerate(zip(self.model_decoders, self.tgt_embeds))
        ]
            
        return dec_outs

  

class EnsembleGenerator(nn.Module):
    """
    Dummy Generator that delegates to individual real Generators,
    and then averages the resulting target distributions.
    """

    def __init__(self, model_generators, raw_probs=False):
        super(EnsembleGenerator, self).__init__()
        self.model_generators = nn.ModuleList(model_generators)
        self._raw_probs = raw_probs

    def forward(self, hidden):#, attn=None, src_map=None):
        """
        Compute a distribution over the target dictionary
        by averaging distributions from models in the ensemble.
        All models in the ensemble must share a target vocabulary.
        """
        distributions = torch.stack(
            [
                mg(h) #if attn is None else mg(h, attn, src_map)
                for h, mg in zip(hidden, self.model_generators)
            ]
        )
        if self._raw_probs:
            return torch.log(torch.exp(distributions).mean(0))
        else:
            return distributions.mean(0)


class EnsembleModel(nn.Module):
    """Dummy NMTModel wrapping individual real NMTModels."""

    def __init__(self, models, raw_probs=False):
        super(EnsembleModel, self).__init__()

        self.encoder = EnsembleEncoder([model.encoder for model in models],[model.src_embed for model in models])
        self.decoder = EnsembleDecoder([model.decoder for model in models],[model.tgt_embed for model in models])
        self.generator = EnsembleGenerator(
            [model.generator for model in models], raw_probs
        )
        self.models = nn.ModuleList(models)

    def forward(self, formula, spec, src_mask, tgt, tgt_mask):
        """
        Take in and process masked src and target sequences.
        src_mask: formula mask + spectra mask
        """
        return self.decode(self.encode(formula, spec, src_mask), tgt, tgt_mask, src_mask)

    def encode(self, formula, spec, src_mask):
        return self.encoder(formula, spec, src_mask)

    def decode(self, memory, tgt, tgt_mask, src_mask):
        return self.decoder(memory, tgt, src_mask, tgt_mask)

def main():
    parser = argparse.ArgumentParser(description='Translate script.')
    parser.add_argument('--para', type=str, help='.yaml file of training parameters')
    args = parser.parse_args()
    with open(args.para, 'r') as f:
        config = yaml.safe_load(f)

    
    if config['spec_embed'] == 'EmbedPatchAttention':
        assert int(config['spec_mask_len']) == int(int(config['spec_len'])/int(config['patch_len'])), "`spec_mask_len` should be `spec_len` divided by `patch_len`."
        spec_embed = EmbedPatchAttention(spec_len=config['spec_len'], patch_len=config['patch_len'], d_model=config['d_model'], src_vocab=100)
    smi_vocab = torch.load(config['vocab_smiles'])
    
    models = []
    for model_i in config['models']:
        
        if config['formula']:
            formula_vocab = torch.load(config['vocab_formula'])
            model = make_model_withFormula(spec_embed=spec_embed, formula_vocab=len(formula_vocab), tgt_vocab=len(smi_vocab),
                                                d_model=config['d_model'], h=config['num_heads'], N=config["layer_num"], d_ff=config["d_ff"], dropout=config["dropout"])
        else:
            model = make_model_onlySpec(tgt_vocab=len(smi_vocab), src_embed=spec_embed, 
                                        N=config["layer_num"], d_ff=config["d_ff"], dropout=config["dropout"])
        model_i_ = torch.load(model_i) 
        if '.tar' in model_i: model.load_state_dict(model_i_['model_state_dict'])
        elif '.pt' in model_i: model.load_state_dict(model_i_)
        models.append(model)
    
    ensemble_model = EnsembleModel(models)#, opt.avg_raw_probs)


    Translate = Ensemble_Translate_Transformer(test_data=config['test_data'], 
                                               save_path=config['save_path'],
                                               vocab_smiles=config['vocab_smiles'],
                                               spec_embed=None, #spec_embed,
                                               spec_mask_len=config['spec_mask_len'],
                                               model=ensemble_model, 
                                               d_model=config['d_model'], 
                                               num_heads=config['num_heads'],
                                               layer_num=config["layer_num"],
                                               d_ff=config["d_ff"],
                                               dropout=config["dropout"],
                                               batchsize=config['batch_size'], 
                                               formula=config['formula'], 
                                               vocab_formula=config['vocab_formula'], 
                                               formula_max_padding=config['formula_max_padding'],
                                               smiles_max_padding=config['smiles_max_padding'],
                                               seed=config['seed'],
                                               )
    
    Translate.translate_main(beam_width=config['beam_width'], 
                             n_best=config['n_best'], 
                             testInf=config['testInf'])

    filename = "BeamSearch_BW{}_NB{}_result.txt".format(config['beam_width'], config['n_best'])
    save_path = config['save_path']
    score_main(inference_path=os.path.join(save_path, filename),
               tgt_path=os.path.join(save_path, 'tgt.txt'),
               save_path=save_path,
               n_beams=config['n_best'])

if __name__ == "__main__":
    main()