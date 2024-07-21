import argparse
import yaml
import os

from inference_process import Translate_Transformer
from .model import EmbedPatchAttention, SpecDirectEmbed
from .utils import score_main

def main():
    parser = argparse.ArgumentParser(description='Translate script.')
    parser.add_argument('--para', type=str, help='.yaml file of training parameters')
    args = parser.parse_args()
    with open(args.para, 'r') as f:
        config = yaml.safe_load(f)


    if config['spec_embed'] == 'EmbedPatchAttention':
        assert int(config['spec_mask_len']) == int(int(config['spec_len'])/int(config['patch_len'])), "`spec_mask_len` should be `spec_len` divided by `patch_len`."
        spec_embed = EmbedPatchAttention(spec_len=config['spec_len'], patch_len=config['patch_len'], d_model=config['d_model'], src_vocab=100)
    elif config['spec_embed'] == 'DirectEmbed':
        spec_embed = SpecDirectEmbed(d_model=config['d_model'], src_vocab=100)

    Translate = Translate_Transformer(test_data=config['test_data'], 
                                    save_path=config['save_path'],
                                    vocab_smiles=config['vocab_smiles'],
                                    spec_embed=spec_embed,
                                    spec_mask_len=config['spec_mask_len'],
                                    model=config['model'], 
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



