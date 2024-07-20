import pandas as pd
import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")
from utils.DatasetDataLoader import generateDataset, split_smiles

import os
import torch
from sklearn.utils import shuffle
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors


def getDataset(all_data, test_data, formula_vocab, smiles_vocab, save_path, smi_max_pad=40):
    # pd.to_pickle(all_data, os.path.join(save_path, 'all_data.pkl'))
    train_data = pd.DataFrame()
    # for i in range(all_data.shape[0]):
    # both_ir_smi = []
    # else_smi = []
    skip_num = 0
    for smi in all_data['smiles']:
        if smi in test_data['smiles'].values: 
            continue
        elif len(split_smiles(smi)) > smi_max_pad:
            skip_num += 1
            print("{} | Skip molecule: {} with mol weight {}".format(skip_num, smi, Descriptors.MolWt(Chem.MolFromSmiles(smi))))
            continue
        train_data = pd.concat([train_data, all_data[all_data['smiles']==smi]],ignore_index=True)
    print(train_data)
    pd.to_pickle(train_data, os.path.join(save_path, 'train_data_smipad40.pkl'))
    
    return

    # data_num = train_data.shape[0]
    # train_val_data = shuffle(train_data)

    # train_data_new = train_val_data[:int(data_num*0.9)]
    # val_data = train_val_data[int(data_num*0.9):]
    # print(train_data_new.shape, val_data.shape)

    # # exp train dataset
    # train_set = generateDataset(train_data_new, smiles_vocab=smiles_vocab, spec_len=3200, formula=True, formula_vocab=formula_vocab, 
    #                             aug_mode=None, aug_num=None, max_shift=None, theta=None, alpha=None)
    # torch.save(train_set, os.path.join(save_path, 'exp_train_set.pt'))
    # val_set = generateDataset(val_data, smiles_vocab=smiles_vocab, spec_len=3200, formula=True, formula_vocab=formula_vocab, 
    #                           aug_mode=None, aug_num=None, max_shift=None, theta=None, alpha=None)
    # torch.save(val_set, os.path.join(save_path, 'exp_val_set.pt'))
    

    # exp test dataset
    test_set = generateDataset(shuffle(test_data), smiles_vocab=smiles_vocab, spec_len=3200, formula=True, formula_vocab=formula_vocab, 
                                aug_mode=None, aug_num=None, max_shift=None, theta=None, alpha=None)
    torch.save(test_set, os.path.join(save_path, 'exp_test_set.pt'))
    # simulated test dataset
    sim_test = test_data.copy().drop(columns=['spectra'])
    sim_test = sim_test.rename({'sim_spectra':'spectra'},axis=1)
    print('sim test dataframe:')
    print(sim_test)
    train_set = generateDataset(sim_test, smiles_vocab=smiles_vocab, spec_len=3200, formula=True, formula_vocab=formula_vocab, 
                                aug_mode=None, aug_num=None, max_shift=None, theta=None, alpha=None)
    torch.save(train_set, os.path.join(save_path, 'sim_test_set.pt'))
    
    
if __name__ == "__main__":
    # pass
    data_savepath = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/experimentalData/data/0withformula_noAug"
    if not os.path.exists(data_savepath): os.mkdir(data_savepath)

    # 781
    test_data = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/experimentalData/data/data_info/bothExpData.pkl")
    # 7225
    all_data = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/experimentalData/data/data_info/validExpData.pkl")

    formula_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/ps8_bs128_EmbedPatchAttention/train1/vocab/vocab_formula.pt")
    smiles_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/ps8_bs128_EmbedPatchAttention/train1/vocab/vocab_smiles.pt")
    
    getDataset(all_data, test_data, formula_vocab, smiles_vocab, save_path=data_savepath)