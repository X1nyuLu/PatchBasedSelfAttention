# come from `modified_ViT.utils.newDatasetDataLoader_withAug`

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset

import logging
from tqdm import tqdm

import pandas as pd
import numpy as np
import random

import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")
# import IRtoMol.scripts.prepare_data as prepdata
from utils.SmilesEnumerator import SmilesEnumerator
from rdkit import Chem

from typing import Dict, List, Optional, Protocol, Tuple
from scipy import interpolate

import re
import os

def split_smiles(smile: str) -> str:
    pattern_full = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern_full)
    tokens = [token for token in regex.findall(smile)]

    if smile != "".join(tokens):
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(tokens, smile)
        )

    return tokens



class generateSMILESDataset(Dataset):
    def __init__(self, data, 
                 smiles_vocab,
                 num_newsmi=5):
        """
        data: '.pkl' file or pandas.Dataframe()
            contains smiles, formula and spectra of interested molecules
        smiles_vocab: torchtext.vocab.Vocab()
        formula_vocab: None or torchtext.vocab.Vocab()
        aug_mode: None or 'verticalNoise' or 'horizontalShift'
        """

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Generate_Dataset")

        if type(data) == str and os.path.isfile(data): 
            assert os.path.exists(data), "Data does not exist: {}".format(data)
            data = pd.read_pickle(data)
        else: data = data

        self.num_newsmi = num_newsmi

        self.logger.info("Generate random SMILES...")
        src_smiles, tgt_smiles = self.getRandomSMILES(data['smiles'])
        self.logger.info("Convert SMILES to tensors...")
        self.src_smi_list, self.src_smi_max_len = self.txt2tensor([split_smiles(i) for i in src_smiles], smiles_vocab)
        self.tgt_smi_list, self.tgt_smi_max_len = self.txt2tensor([split_smiles(i) for i in tgt_smiles], smiles_vocab)

    def txt2tensor(self, txt_split_list, vocab):
        new_lines = []
        max_len = 0
        for line in tqdm(txt_split_list):
            line = torch.tensor([ vocab[token] for token in line])
            if line.shape[0] > max_len:
                max_len = line.shape[0]
            new_lines.append(line)

        return new_lines, max_len

    def getRandomSMILES(self, smiles_ori):
        src_smiles = [] # random smiles
        tgt_smiles = [] # canolical smiles
        sme = SmilesEnumerator()
        for smi in smiles_ori:
            src_smiles += [sme.randomize_smiles(smi) for j in range(self.num_newsmi)]
            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol)
            tgt_smiles += [can_smi]*self.num_newsmi
        return src_smiles, tgt_smiles
    
    def __len__(self):
        return len(self.tgt_smi_list)

    def __getitem__(self, index):
        src_smi = self.src_smi_list[index]
        tgt_smi = self.tgt_smi_list[index]

        return src_smi, tgt_smi


class CreateSMILESDataloader:
    def __init__(self, pad_id, bs_id, eos_id, 
                 src_max_padding, tgt_max_padding):
        self.pad_id = pad_id
        self.bs_id = bs_id
        self.eos_id = eos_id
        self.src_max_pad = src_max_padding
        self.tgt_max_pad = tgt_max_padding
        
    
    def collate_batch(self, batch):
        bs_id = torch.tensor([self.bs_id])  # <s> token id
        eos_id = torch.tensor([self.eos_id])  # </s> token id
        
        src_smi_list = []
        tgt_smi_list = []

        for (_src, _tgt) in batch:
            processed_src = torch.cat([bs_id, _src, eos_id], 0)
            src_smi_list.append( pad(processed_src, (0, self.src_max_pad - len(processed_src)), value=self.pad_id) )

            processed_tgt = torch.cat([bs_id, _tgt, eos_id], 0)
            tgt_smi_list.append( pad(processed_tgt, (0, self.tgt_max_pad - len(processed_tgt)), value=self.pad_id) )

        src_smi = torch.stack(src_smi_list)
        tgt_smi = torch.stack(tgt_smi_list)

       
        return {"src_smi": src_smi, "tgt_smi": tgt_smi}
        

    def dataloader(self,
        dataset,
        batch_size,
        shuffle=True,
    ):
       

        def collate_fn(batch):
            return self.collate_batch(batch)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=True
        )
        return dataloader


def set_random_seed(seed):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for Random Shuffler of batches
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        # This one is needed for various tranfroms
        np.random.seed(seed)


if __name__ == "__main__":
    # pass
    # formula_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_formula.pt")
    smiles_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_smiles.pt")
    data = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/30data.pkl"
    dataset = generateSMILESDataset(data=data,smiles_vocab=smiles_vocab, num_newsmi=5)
    # print(len(dataset))
    # # print(dataset)
    # src, tgt = dataset[8]
    # print(src)
    # print(tgt)
    # dataloder_ = CreateSMILESDataloader(pad_id=2, bs_id=0, eos_id=1, src_max_padding=415, tgt_max_padding=40)
    # dataloder = dataloder_.dataloader(dataset,batch_size=10,shuffle=True)
    # for batch in dataloder:
    #     # print(batch)
    #     print(batch['src_smi'][0])
    #     print(batch['tgt_smi'][0])
    #     break


    smiles = ["OC1CCCCC1n1cnc2cncnc21"]
    src_smiles, tgt_smiles = dataset.getRandomSMILES(smiles)
    print('src')
    print(src_smiles)
    print('tgt')
    print(tgt_smiles)
    # print(prepdata.split_smiles(smiles))
    # print(split_smiles(smiles))

    # f = "C11H14N4O"
    # print(prepdata.split_formula(f))
    # print(split_formula(f))
    pass