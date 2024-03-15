import yaml
# import sys
from train_smilesModel_process import TrainVal


import argparse

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--train_para', type=str, help='.yaml file of training parameters')
args = parser.parse_args()
print('arg para: ', args.train_para)

with open(args.train_para, 'r') as f:
    config = yaml.safe_load(f)


TRAIN = TrainVal(data=config['data'], 
                 save_path=config['save_path'],
                 vocab_smiles=config['vocab_smiles'],
                 num_newsmi=config['num_newsmi'],
                 model=config['model'], 
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
                 src_max_pad=config['src_max_padding'],
                 tgt_max_pad=config['tgt_max_padding'],
                 testTrain=config['test_train'], 
                 seed=config['seed'],
                 earlystop_delta=config['earlystop_delta'],
                 earlystop_patience=config['earlystop_patience']
                 )
                 
TRAIN.train_worker()

