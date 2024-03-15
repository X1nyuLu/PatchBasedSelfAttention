import yaml
from translate_smilesModele_process import Translate_Transformer
from model.spectra_process_layer import *
from utils.score import main as score_main
import os

import argparse

parser = argparse.ArgumentParser(description='Translate script.')
parser.add_argument('--translate_para', type=str, help='.yaml file of training parameters')
args = parser.parse_args()
print('arg para: ',args.translate_para)
with open(args.translate_para, 'r') as f:
    config = yaml.safe_load(f)


Translate = Translate_Transformer(test_data=config['test_data'], 
                                  save_path=config['save_path'],
                                  vocab_smiles=config['vocab_smiles'],
                                  model=config['model'], 
                                  d_model=config['d_model'], 
                                  num_heads=config['num_heads'],
                                  layer_num=config["layer_num"],
                                  d_ff=config["d_ff"],
                                  dropout=config["dropout"],
                                  batchsize=config['batch_size'], 
                                  src_max_pad=config["src_max_pad"],
                                  tgt_max_pad=config["tgt_max_pad"],
                                  seed=config['seed'],
                                  )
Translate.translate_main(mode=config['translate_mode'], 
                         beam_width=config['beam_width'], 
                         n_best=config['n_best'], 
                         testInf=config['testInf'])
# """
mode = config['translate_mode']
if mode == "G": filename = "GreedySearch_result.txt"
elif mode == "B": filename = "BeamSearch_BW{}_NB{}_result.txt".format(config['beam_width'], config['n_best'])
save_path = config['save_path']
score_main(inference_path=os.path.join(save_path, filename),
           tgt_path=os.path.join(save_path, 'tgt.txt'),
           save_path=save_path,
           n_beams=config['n_best'])


