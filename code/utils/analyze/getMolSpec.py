import pandas as pd
import torch
import os
import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")
from utils.DatasetDataLoader import generateDataset

mol_list_f = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/analyze/mole_cls_result/mol/mol_list.txt"
save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/analyze/mole_cls_result/mol"
with open(mol_list_f) as f:
    mol_list = f.readlines()
mol_list = [mol.split('\n')[0] for mol in mol_list]
print(mol_list)
data = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/IRtoMol/data/ir_data.pkl")
data_ = data[data['smiles'].isin(mol_list)]
print(data_)
pd.to_pickle(data_, os.path.join(save_path,"mol_spc.pkl"))

# data_ = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/analyze/mole_cls_result/mol_spc.pkl")
vocab_formula= "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/ps8_bs128_EmbedPatchAttention/train1/vocab/vocab_formula.pt"
vocab_smiles= "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/ps8_bs128_EmbedPatchAttention/train1/vocab/vocab_smiles.pt"
spec_len= 3200
patch_len= 64
spec_mask_len= 50
formula= True
formula_max_padding= 15
smiles_max_padding= 30
data_set = generateDataset(data=data_,smiles_vocab=torch.load(vocab_smiles),spec_len=spec_len, 
                           formula=formula, formula_vocab=torch.load(vocab_formula))
torch.save(data_set, os.path.join(save_path,"mol_dataset.pt"))
