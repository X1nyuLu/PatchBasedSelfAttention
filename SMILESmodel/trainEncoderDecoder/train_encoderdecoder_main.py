import yaml
import sys
sys.path.append('/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code')
from train_process import TrainVal
from model.spectra_process_layer import *
from model.formula_spec_model import make_model as make_model_withFormula
from model.spec_model import make_model as make_model_onlySpec


import argparse

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--train_para', type=str, help='.yaml file of training parameters')
args = parser.parse_args()
with open(args.train_para, 'r') as f:
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

spec_checkpoint = torch.load(config['spec_model'])
spec_model_state_dict_to_load = spec_checkpoint['model_state_dict']
encoder_state_dict = {k: v for k, v in spec_model_state_dict_to_load.items() if ('encoder' in k or ('src' in k))}

smi_checkpoint = torch.load(config['smi_model'])
smi_model_state_dict_to_load = smi_checkpoint['model_state_dict']
decoder_state_dict = {k: v for k, v in smi_model_state_dict_to_load.items() if (('decoder' in k) or ('generator' in k) or ('tgt' in k))}

merged_dict = encoder_state_dict | decoder_state_dict
model.load_state_dict(merged_dict, strict=True)

if config['fix_encoder']:
    for name, param in model.named_parameters():
        if ("encoder" in name) or ("src" in name) :
            param.requires_grad = False
elif config['fix_decoder']:
    for name, param in model.named_parameters():
        if ("decoder" in name) or ("tgt" in name) or ("generator" in name):
            param.requires_grad = False


TRAIN = TrainVal(data=config['data'], 
                 save_path=config['save_path'],
                 vocab_smiles=config['vocab_smiles'],
                 spec_embed=None,
                 spec_len=config['spec_len'],
                 spec_mask_len=config['spec_mask_len'],
                 model=model, 
                 d_model=config['d_model'], 
                 num_heads=config['num_heads'], 
                 layer_num=config["layer_num"],
                 d_ff=config["d_ff"],
                 dropout=config["dropout"],
                 batchsize=config['batch_size'], 
                 accum_iter=config['accum_iter'], 
                 num_epochs=config['num_epochs'], 
                 warmup=config['warmup_steps'], 
                 base_lr=config['base_lr'],
                 Resume=config['Resume'],
                 aug_mode=config['aug_mode'], 
                 testset_aug=config['testset_aug'],
                 formula=config['formula'], 
                 vocab_formula=config['vocab_formula'], 
                 formula_max_padding=config['formula_max_padding'],
                 smiles_max_padding=config['smiles_max_padding'],
                 testTrain=config['test_train'], 
                 seed=config['seed'],
                 earlystop_delta=config['earlystop_delta'],
                 earlystop_patience=config['earlystop_patience']
                 )
                 
TRAIN.train_worker()

