import yaml
# import sys
# sys.path.append('/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult')
from train_encoder_process import TrainVal
from model.spectra_process_layer import *


import argparse

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--train_para', type=str, help='.yaml file of training parameters')
args = parser.parse_args()
with open(args.train_para, 'r') as f:
    config = yaml.safe_load(f)

# learning_rate = config['learning_rate']
# batch_size = config['batch_size']
# epochs = config['epochs']
if config['spec_embed'] == 'EmbedPatchAttention':
    assert int(config['spec_mask_len']) == int(int(config['spec_len'])/int(config['patch_len'])), "`spec_mask_len` should be `spec_len` divided by `patch_len`."
    spec_embed = EmbedPatchAttention(spec_len=config['spec_len'], patch_len=config['patch_len'], d_model=config['d_model'], src_vocab=100)


TRAIN = TrainVal(data=config['data'], 
                 save_path=config['save_path'],
                 vocab_smiles=config['vocab_smiles'],
                 spec_embed=spec_embed,
                 spec_len=config['spec_len'],
                 spec_mask_len=config['spec_mask_len'],
                 spec_model=config['spec_model'], smi_model=config['smi_model'],
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

