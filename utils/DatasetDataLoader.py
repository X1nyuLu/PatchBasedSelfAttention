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
# sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/")
# import IRtoMol.scripts.prepare_data as prepdata
from typing import Dict, List, Optional, Protocol, Tuple
from scipy import interpolate
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")
from utils.SmilesEnumerator import SmilesEnumerator
from rdkit import Chem

import re
import os

def split_smiles(smile: str) -> list:
    pattern_full = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern_full)
    tokens = [token for token in regex.findall(smile)]

    if smile != "".join(tokens):
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(tokens, smile)
        )

    return tokens

def split_formula(formula: str) -> list:
    segments = re.findall(r"[A-Z][a-z]*\d*", formula)
    formula_split = [a for segment in segments for a in re.split(r"(\d+)", segment)]
    formula_split = list(filter(None, formula_split))

    if "".join(formula_split) != formula:
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(
                formula_split, formula
            )
        )
    # return " ".join(formula_split)
    return formula_split


def norm_spectrum(
    spectrum: np.ndarray, bounds: Tuple[int, int] = (0, 99)
) -> np.ndarray:
    spectrum_norm = spectrum / max(spectrum) * bounds[1]
    spectrum_norm_int = spectrum_norm.astype(int)
    spectrum_norm_int = np.clip(spectrum_norm_int, *bounds)

    return spectrum_norm_int

def interpolate_spectrum(
    spectrum: np.ndarray,
    new_x: np.ndarray,
    orig_x: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    if orig_x is None:
        orig_x = np.arange(400, 3982, 2)

    intp = interpolate.interp1d(orig_x, spectrum)

    intp_spectrum = intp(new_x)
    intp_spectrum_norm = norm_spectrum(intp_spectrum)

    return [intp_spectrum_norm]

def augment_shift_horizontal_orig(spectrum: np.ndarray, new_x: np.ndarray) -> List[np.ndarray]:
    orig_x = np.arange(400, 3982, 4)
    set1 = spectrum[::2] # orig spec
    set2 = np.concatenate([spectrum[1::2], np.reshape(set1[-1], 1)]) # shift 1 unit

    aug_spec1 = interpolate_spectrum(set1, new_x, orig_x)[0]
    aug_spec2 = interpolate_spectrum(set2, new_x, orig_x)[0]
    return [aug_spec1, aug_spec2]

class generateDataset(Dataset):
    def __init__(self, data, 
                 smiles_vocab, #smiles_max_pad=40,
                 spec_len=3200,
                 formula=False, formula_vocab=None, #formula_max_pad=15,
                 aug_mode=None, aug_num=0, smi_aug_num=0,
                 max_shift=None, theta=0.01, alpha=1
                 ):
        """
        data: '.pkl' file or pandas.Dataframe()
            contains smiles, formula and spectra of interested molecules
        smiles_vocab: torchtext.vocab.Vocab()
        formula_vocab: None or torchtext.vocab.Vocab()
        aug_mode: None or 'verticalNoise' or 'horizontalShift' or 'horizontalShiftNonFP' or 'SMILES'
        """

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Generate_Dataset")
        self.formula = formula


        if type(data) == str and os.path.isfile(data): 
            assert os.path.exists(data), "Data does not exist: {}".format(data)
            data = pd.read_pickle(data)
        else: data = data

        # print(type(aug_mode))
        if type(aug_mode) == list:
            for i in aug_mode:
                assert i in [ None, 'verticalNoise', 'horizontalShift', 'horizontalShiftNonFP', 'SMILES'], "Please check aug_mode set"
        elif type(aug_mode) == str:
            assert aug_mode in [ None, 'verticalNoise', 'horizontalShift', 'horizontalShiftNonFP', 'SMILES'], "Please check aug_mode set"

        self.logger.info("Dealing with input of SPECTRA...")
        self.spec_list = self.process_spec(spec_len, data["spectra"], aug_mode=aug_mode, aug_num=aug_num, smi_aug_num=smi_aug_num,
                                           max_shift=max_shift, theta=theta, alpha=alpha)

        self.logger.info("Dealing with input of SMILES...")
        self.smi_list, self.smi_max_len = self.smiles2tensor(data["smiles"], smiles_vocab, #smi_max_pad=smiles_max_pad,
                                                             aug_mode=aug_mode, aug_num=aug_num, smi_aug_num=smi_aug_num)
        self.logger.info("SMILES max len: {}".format(self.smi_max_len))

        if self.formula:
            self.logger.info("Dealing with input of FORMULAS...")
            self.formula_list, self.formula_max_len = self.formula2tensor(data["formula"], formula_vocab, #formula_max_pad=formula_max_pad,
                                                                          aug_mode=aug_mode, aug_num=aug_num, smi_aug_num=smi_aug_num)
            self.logger.info("FORMULA max len: {}".format(self.formula_max_len))


    def process_spec(self, spec_len, spec_data, aug_mode, aug_num, max_shift=None, theta=None, alpha=None, smi_aug_num=0):
        new_x = np.linspace(400, 3980, spec_len)
        if aug_mode is None:
            spec_list = [torch.tensor(interpolate_spectrum(i, new_x)[0]) for i in spec_data]

        elif aug_mode == 'verticalNoise' or aug_mode == ['verticalNoise']:
            assert ((theta!=None) and (alpha!=None)), 'theta({}) and alpha({}) should be set.'.format(theta, alpha)
            spec_list = []
            for spec in spec_data:
                spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x)[0])) # original spec
                for i in range(aug_num):
                    spec_list.append(self.vertical_noise(new_x, spec, theta=theta, alpha=alpha))
                
        elif aug_mode == 'horizontalShift' or aug_mode == ['horizontalShift']:
            spec_list = []
            for spec in spec_data:
                spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x)[0])) # original spec
                
                shift = np.arange(-max_shift, max_shift + 1)
                shift = shift[shift != 0] # shift distance != 0
                assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
                shift = np.random.choice(shift, size=aug_num, replace=False)
                
                for shift_i in shift:
                    spec_list.append(self.shift_horizontal(spec, shift=shift_i, new_x=new_x))
                    
        elif aug_mode == 'horizontalShiftNonFP':
            spec_list = []
            for spec in spec_data:
                spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x)[0])) # original spec
                
                shift = np.arange(-max_shift, max_shift + 1)
                shift = shift[shift != 0]
                assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
                shift = np.random.choice(shift, size=aug_num, replace=False)
                
                for shift_i in shift:
                    spec_list.append(self.shift_horizontal_nonFP(spec, shift=shift_i, new_x=new_x))
        elif aug_mode == 'SMILES' or aug_mode == ['SMILES'] :
            spec_list = []
            for spec in spec_data:
                # spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] * (aug_num + 1)
                spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] * (smi_aug_num + 1)
                # for i in range(aug_num):
        
        # elif set(aug_mode) == set(['verticalNoise', 'horizontalShift']): 
        elif ('verticalNoise' in aug_mode) and ('horizontalShift' in aug_mode):
            assert ((theta!=None) and (alpha!=None)), 'theta({}) and alpha({}) should be set.'.format(theta, alpha)
            assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
            
            shift = np.arange(-max_shift, max_shift + 1)
            shift = shift[shift != 0] # shift distance != 0
            shift = np.random.choice(shift, size=aug_num, replace=False)
            
            spec_list = []
            for spec in spec_data:
                if set(aug_mode) == set(['verticalNoise', 'horizontalShift']):
                    spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x)[0])) # original spec
                elif set(aug_mode) == set(['verticalNoise', 'horizontalShift', 'SMILES']):
                    # spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] * (aug_num + 1) # original spec
                    spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] * (smi_aug_num + 1) # original spec
                
                for i in range(aug_num):
                    spec_i = self.vertical_noise(new_x, spec, theta=theta, alpha=alpha)
                    spec_i = self.shift_horizontal(spec_i, shift=shift[i], new_x=new_x, orig_x=new_x)
                    if set(aug_mode) == set(['verticalNoise', 'horizontalShift']):
                        spec_list.append(spec_i)
                    elif set(aug_mode) == set(['verticalNoise', 'horizontalShift', 'SMILES']):
                        # spec_list += [spec_i] * (aug_num + 1)
                        spec_list += [spec_i] * (smi_aug_num + 1)
        
        elif set(aug_mode) == set(['verticalNoise', 'SMILES']): 
            assert ((theta!=None) and (alpha!=None)), 'theta({}) and alpha({}) should be set.'.format(theta, alpha)
            spec_list = []
            for spec in spec_data:
                # spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] # original spec
                # spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] * (aug_num + 1) # original spec
                spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] * (smi_aug_num + 1) # original spec
                # print()
                # print(len(spec_list))
                for i in range(aug_num):
                    # spec_list.append(self.vertical_noise(new_x, spec, theta=theta, alpha=alpha))
                    # spec_list += [self.vertical_noise(new_x, spec, theta=theta, alpha=alpha)] * (aug_num + 1)
                    spec_list += [self.vertical_noise(new_x, spec, theta=theta, alpha=alpha)] * (smi_aug_num + 1)
                    # print(len(spec_list))
        elif set(aug_mode) == set(['horizontalShift', 'SMILES']): 
            spec_list = []
            for spec in spec_data:
                spec_list += [torch.tensor(interpolate_spectrum(spec, new_x)[0])] * (smi_aug_num + 1)# original spec
                
                shift = np.arange(-max_shift, max_shift + 1)
                shift = shift[shift != 0] # shift distance != 0
                assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
                shift = np.random.choice(shift, size=aug_num, replace=False)
                
                for shift_i in shift:
                    spec_list += [self.shift_horizontal(spec, shift=shift_i, new_x=new_x)] * (smi_aug_num + 1)
            
        return spec_list
        
    
    def vertical_noise(self, spec_x, spec_y, theta=0.01, alpha=1, orig_x=None):
        """
        theta: threshold of points which will have noise
        alpha: extent of noise.
        """
        spec_y = spec_y/max(spec_y) # y value of the highest peak is 1.

        random_values = np.zeros(spec_y.shape)
        indices =  spec_y > theta # Only add noise on peaks, in case it changed the meaning of the sepctrum

        # Uniformly sample random number between [-1, alpha-1)
        random_values[indices] = (np.random.rand(sum(indices))*alpha - 1)
        aug_y = spec_y * (1 + random_values)
        return interpolate_spectrum(aug_y, spec_x, orig_x)[0]

    def shift_horizontal(self, spectrum, shift, new_x, orig_x=None):
        if orig_x is None:
            orig_x = np.arange(400, 3982, 2)
        shifted_x = orig_x + shift
        interp_func = interpolate.interp1d(shifted_x, spectrum, bounds_error=False, fill_value=0)
        shifted_spectrum = interp_func(new_x)
        return norm_spectrum(shifted_spectrum)
        # return
    
    def shift_horizontal_nonFP(self, spectrum, shift, new_x, threshold=1500):
        """
        Only shift non-fingerprint area (x>1500cm-1)
        """
        orig_x = np.arange(400, 3982, 2)

        # split the spectrum
        mask = orig_x > threshold
        x_right = orig_x[mask]
        spectrum_right = spectrum[mask]
        
        # shift the part of x > threshold
        x_right_shifted = x_right + shift
        
        # create a spectrum after shift
        interp_func = interpolate.interp1d(x_right_shifted, spectrum_right, bounds_error=False, fill_value=0)
        
        # ensure the spectrum's continuity 
        spectrum_shifted = np.where(orig_x <= threshold, spectrum, interp_func(orig_x))

        
        # return norm_spectrum(spectrum_shifted)
        return interpolate_spectrum(spectrum_shifted, new_x, orig_x)[0]

    def smiles2tensor(self, smi_data, vocab, aug_mode, aug_num, smi_aug_num=0):
        new_lines = []
        max_len = 0
        for smi in tqdm(smi_data):
            len_list = []

            smi_ori = torch.tensor([ vocab[token] for token in split_smiles(smi)])
            len_list.append(smi_ori.shape[0])
            # print(smi)
            if aug_mode is None: new_lines += [smi_ori]
            elif aug_mode == 'SMILES' or 'SMILES' in aug_mode:
                smi_tensor_list = [smi_ori]                

                sme = SmilesEnumerator()
                aug_smi_list = []
                j = 0
                smi_tryNum = {}
                # while j < aug_num:
                while j < smi_aug_num:
                    smi_j = sme.randomize_smiles(smi)

                    # The number of some molecules' SMILES are less than aug num, like C1CCCC1, COC.
                    if smi_j in aug_smi_list: 
                        if smi_j in smi_tryNum: 
                            # if smi_tryNum[smi_j] > aug_num: pass
                            if smi_tryNum[smi_j] > smi_aug_num: pass
                            else: 
                                smi_tryNum[smi_j] += 1
                                continue
                        else: 
                            smi_tryNum[smi_j] = 1
                            continue
                        # if len(set(aug_smi_list)) == 1: pass 
                        # else: 
                            # continue
                    aug_smi_list.append(smi_j)

                    smi_j_ = torch.tensor([ vocab[token] for token in split_smiles(smi_j)])
                    len_list.append(smi_j_.shape[0])
                    smi_tensor_list.append(smi_j_)

                    j += 1
                
                # new_lines += smi_tensor_list
                
                if aug_mode =='SMILES':
                    new_lines += smi_tensor_list
                elif type(aug_mode) == list:
                    # new_lines += smi_tensor_list * (len(aug_mode)-1) * (aug_num+1)
                    new_lines += smi_tensor_list * (aug_num+1)

            # elif aug_mode == 'verticalNoise' or aug_mode == 'horizontalShift' or aug_mode=='horizontalShiftNonFP':
            # elif aug_mode != None:
            else:
                new_lines += [smi_ori] * (aug_num+1)
            # print(new_lines)
            if max(len_list) > max_len:
                max_len = max(len_list)
            
        return new_lines, max_len
    
    def formula2tensor(self, formula_data, vocab, aug_mode, aug_num, smi_aug_num):
        txt_split_list = [split_formula(i) for i in formula_data]
        new_lines = []
        max_len = 0
        for line_ in tqdm(txt_split_list):
            line = torch.tensor([ vocab[token] for token in line_])
            
            if line.shape[0] > max_len:
                max_len = line.shape[0]
            

            # if aug_mode == 'verticalNoise' or aug_mode == 'horizontalShift' or aug_mode=='horizontalShiftNonFP' or aug_mode=='SMILES':
            if aug_mode is None: new_lines.append(line)
            elif type(aug_mode) == str or len(aug_mode) == 1:
                if aug_mode == 'SMILES':
                    new_lines += [line] * (smi_aug_num+1)
                else:
                    new_lines += [line] * (aug_num+1)
            elif type(aug_mode) == list:
                if 'SMILES' in aug_mode:
                    # new_lines += [line] * (len(aug_mode)-1) * (aug_num+1)**2
                    new_lines += [line] * (aug_num+1) * (smi_aug_num+1)
                else: new_lines += [line] * (aug_num+1)
        return new_lines, max_len
    
    def __len__(self):
        assert len(self.spec_list) == len(self.smi_list), "The number of spectra data({}) does not equal to that of SMILES data({}).".format(len(self.spec_list),len(self.smi_list))
        assert len(self.spec_list) == len(self.formula_list), "The number of spectra data({}) does not equal to that of formula data({}).".format(len(self.spec_list), len(self.formula_list))
        return len(self.spec_list)

    def __getitem__(self, index):
        # print("index: ", index)
        spec = self.spec_list[index]
        smi = self.smi_list[index]

        if self.formula:
            formula = self.formula_list[index]
            return (spec, formula), smi
        else: return spec, smi



class CreateDataloader:
    def __init__(self, pad_id, bs_id, eos_id, 
                 tgt_max_padding, 
                 formula=False, formula_max_pad=None):
        self.pad_id = pad_id
        self.bs_id = bs_id
        self.eos_id = eos_id
        self.tgt_max_pad = tgt_max_padding
        
        self.formula = formula
        self.formula_max_pad = formula_max_pad
    
    def collate_batch(self, batch):
        bs_id = torch.tensor([self.bs_id])  # <s> token id
        eos_id = torch.tensor([self.eos_id])  # </s> token id
        
        spec_list = []
        smi_list = []
        if self.formula: formula_list = []

        for (_src, _smi) in batch:
            processed_smi = torch.cat([bs_id, _smi, eos_id], 0)
            
            smi_list.append( pad(processed_smi, (0, self.tgt_max_pad + 2 - len(processed_smi)), value=self.pad_id) )

            if self.formula:
                spec, formula = _src
                processed_formula = torch.cat([bs_id, formula, eos_id], 0)
                formula_list.append( pad(processed_formula, (0, self.formula_max_pad - len(processed_formula)), value=self.pad_id) )

                spec = torch.as_tensor(spec, dtype=torch.float)
                
            else:
                spec = torch.as_tensor(_src, dtype=torch.float)

                
            spec_list.append(spec)

        smi = torch.stack(smi_list)
        spec = torch.stack(spec_list)

        if self.formula:
            formula = torch.stack(formula_list)
            return {"spec": spec, "formula": formula, "smi": smi}
        else: return {"spec": spec, "smi": smi}
        

    def dataloader(self,
        dataset,
        batch_size,
        shuffle,
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
    """
    formula_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_formula.pt")
    smiles_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_smiles.pt")
    data = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/30data.pkl"

    # dataset = generateDataset(data,smiles_vocab, )
    dataset = generateDataset(data,smiles_vocab, 
                            #   formula=True, formula_vocab=formula_vocab,
                            #   aug_mode="horizontalShift", aug_num=3, max_shift=500,
                              aug_mode="horizontalShiftNonFP", aug_num=3, max_shift=500
                              )
    data_df = pd.read_pickle(data)
    # spec = data_df['spectra'][0]
    # print(spec)
    import matplotlib.pyplot as plt
    # x = np.arange(400,3982,2)
    # plt.figure()
    # plt.plot(x, spec, 'b-')
    # # plt.plot(x, dataset.augment_shift_horizontal(spec, max_shift=-100), 'y-')
    # plt.plot(x, dataset.shift_spectrum_partially(spec, shift=-50), 'y-')
    # plt.savefig('test_hori_partial_aug_50.png')
    dataloader_ = CreateDataloader(2,0, 1,40,
                                #    formula=True,formula_max_pad=15
                                   )
    dataloader = dataloader_.dataloader(dataset, 3, shuffle=False)
    save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/aug_test"
    for batch in dataloader:
        # print(batch)
        spec = batch['spec']
        # x = np.arange(400,3982,2)
        x = np.linspace(400, 3980, 3200)

        # plt.figure()
        # plt.plot(x, spec[0], 'b-')
        # plt.plot(x, spec[1], 'y-')
        # plt.plot(x, spec[2], 'g-')
        # plt.savefig(os.path.join(save_path,'test_hori_partial_aug_3.png'))
        break
    """
    # a = 'C=C'
    # print(len(split_smiles(a)))
    pass