import os
import yaml
import argparse
import torch

from ..inference_process import Translate_Transformer
from ..utils import score_main
from ..model import EmbedPatchAttention, make_model_withFormula, make_model_onlySpec



parser = argparse.ArgumentParser(description='Translate script.')
parser.add_argument('--para', type=str, help='.yaml file of training parameters')
args = parser.parse_args()

with open(args.para, 'r') as f:
    config = yaml.safe_load(f)


if config['spec_embed'] == 'EmbedPatchAttention':
    assert int(config['spec_mask_len']) == int(int(config['spec_len'])/int(config['patch_len'])), "`spec_mask_len` should be `spec_len` divided by `patch_len`."
    spec_embed = EmbedPatchAttention(spec_len=config['spec_len'], patch_len=config['patch_len'], d_model=config['d_model'], src_vocab=100)

smi_vocab = torch.load(config['vocab_smiles'])

if config['formula']:
    formula_vocab = torch.load(config['vocab_formula'])
    model = make_model_withFormula(spec_embed=spec_embed, formula_vocab=len(formula_vocab), tgt_vocab=len(smi_vocab),
                                        d_model=config['d_model'], h=config['num_heads'], N=config["layer_num"], d_ff=config["d_ff"], dropout=config["dropout"])
else:
    model = make_model_onlySpec(tgt_vocab=len(smi_vocab), src_embed=spec_embed, 
                                N=config["layer_num"], d_ff=config["d_ff"], dropout=config["dropout"])


model_num = len(config['models'])
assert model_num >= 2, "There is only {} model.".format(model_num)

models_sd = [] # store state_dict
for model_i in config['models']:
    model_i_sd = torch.load(model_i) 
    if '.tar' in model_i: models_sd.append(model_i_sd['model_state_dict'])
    elif '.pt' in model_i: models_sd.append(model_i_sd)

for key in models_sd[0]:
    models_sd[1][key] = sum([models_sd[i][key] for i in range(model_num)])/model_num

model.load_state_dict(models_sd[1])
torch.save(model.state_dict(),os.path.join(config['save_path'],'avg_of_{}_models.pt'.format(model_num)))

Translate = Translate_Transformer(test_data=config['test_data'], 
                                  save_path=config['save_path'],
                                  vocab_smiles=config['vocab_smiles'],
                                  spec_embed=spec_embed,
                                  spec_mask_len=config['spec_mask_len'],
                                  model=model, 
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
