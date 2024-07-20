# come from `modified_ViT.utils.newDatasetDataLoader_withAug`

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import logging
import random
import re
import os

from typing import List, Optional, Tuple
from scipy import interpolate
from tqdm import tqdm

from utils.SmilesEnumerator import SmilesEnumerator


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
    orig_x: Optional[np.ndarray] 
) -> List[np.ndarray]:
    
    intp = interpolate.interp1d(orig_x, spectrum)

    intp_spectrum = intp(new_x)
    intp_spectrum_norm = norm_spectrum(intp_spectrum)

    return [intp_spectrum_norm]


class generateDataset(Dataset):
    def __init__(self, data, 
                 smiles_vocab, 
                 spec_len=3200,
                 formula=False, formula_vocab=None, 
                 aug_mode=None, aug_num=0, smi_aug_num=0,
                 max_shift=None, theta=0.01, alpha=1,
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
        
        orig_x = np.arange(400, 3982, 2)
        new_x = np.linspace(400, 3980, spec_len)

        if aug_mode is None:
            spec_list = [torch.tensor(interpolate_spectrum(i, new_x, orig_x)[0]) for i in spec_data]

        elif aug_mode == 'verticalNoise' or aug_mode == ['verticalNoise']:
            assert ((theta!=None) and (alpha!=None)), 'theta({}) and alpha({}) should be set.'.format(theta, alpha)
            spec_list = []

            for spec in tqdm(spec_data, desc="spectra | {}".format(str(aug_mode))):
                spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x, orig_x)[0])) # original spec
                for i in range(aug_num):
                    spec_list.append(self.vertical_noise(new_x, spec, theta=theta, alpha=alpha, orig_x=orig_x))
                
        elif aug_mode == 'horizontalShift' or aug_mode == ['horizontalShift']:
            spec_list = []
            for spec in tqdm(spec_data, desc="spectra | {}".format(str(aug_mode))):
                spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x, orig_x)[0])) # original spec
                
                shift = np.arange(-max_shift, max_shift + 1)
                shift = shift[shift != 0] # shift distance != 0
                assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
                shift = np.random.choice(shift, size=aug_num, replace=False)
                
                for shift_i in shift:
                    spec_list.append(self.shift_horizontal(spec, shift=shift_i, new_x=new_x, orig_x=orig_x))
                    
        elif aug_mode == 'horizontalShiftNonFP':
            spec_list = []
            for spec in spec_data:
                spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x)[0])) # original spec
                
                shift = np.arange(-max_shift, max_shift + 1)
                shift = shift[shift != 0]
                assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
                shift = np.random.choice(shift, size=aug_num, replace=False)
                
                for shift_i in shift:
                    spec_list.append(self.shift_horizontal_nonFP(spec, shift=shift_i, new_x=new_x, orig_x=orig_x))
        elif aug_mode == 'SMILES' or aug_mode == ['SMILES'] :
            spec_list = []
            assert smi_aug_num != None and smi_aug_num > 0, 'smi_aug_num({}) should be larger than 0.'.format(smi_aug_num)
            for spec in tqdm(spec_data, desc="spectra | {}".format(str(aug_mode))):
                spec_list += [torch.tensor(interpolate_spectrum(spec, new_x, orig_x)[0])] * (smi_aug_num + 1)
        
        elif ('verticalNoise' in aug_mode) and ('horizontalShift' in aug_mode):
            assert ((theta!=None) and (alpha!=None)), 'theta({}) and alpha({}) should be set.'.format(theta, alpha)
            assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
            
            shift = np.arange(-max_shift, max_shift + 1)
            shift = shift[shift != 0] # shift distance != 0
            shift = np.random.choice(shift, size=aug_num, replace=False)
            
            spec_list = []
            for spec in tqdm(spec_data, desc="spectra | {}".format(str(aug_mode))):
                if set(aug_mode) == set(['verticalNoise', 'horizontalShift']):
                    spec_list.append(torch.tensor(interpolate_spectrum(spec, new_x, orig_x)[0])) # original spec
                elif set(aug_mode) == set(['verticalNoise', 'horizontalShift', 'SMILES']):
                    spec_list += [torch.tensor(interpolate_spectrum(spec, new_x, orig_x)[0])] * (smi_aug_num + 1) # original spec
                
                for i in range(aug_num):
                    spec_i = self.vertical_noise(new_x, spec, theta=theta, alpha=alpha, orig_x=orig_x)
                    spec_i = self.shift_horizontal(spec_i, shift=shift[i], new_x=new_x, orig_x=new_x)
                    if set(aug_mode) == set(['verticalNoise', 'horizontalShift']):
                        spec_list.append(spec_i)
                    elif set(aug_mode) == set(['verticalNoise', 'horizontalShift', 'SMILES']):
                        spec_list += [spec_i] * (smi_aug_num + 1)
        
        elif set(aug_mode) == set(['verticalNoise', 'SMILES']): 
            assert ((theta!=None) and (alpha!=None)), 'theta({}) and alpha({}) should be set.'.format(theta, alpha)
            spec_list = []
            for spec in tqdm(spec_data, desc="spectra | {}".format(str(aug_mode))):
                spec_list += [torch.tensor(interpolate_spectrum(spec, new_x, orig_x)[0])] * (smi_aug_num + 1) # original spec
                for i in range(aug_num):
                    spec_list += [self.vertical_noise(new_x, spec, theta=theta, alpha=alpha, orig_x=orig_x)] * (smi_aug_num + 1)
        elif set(aug_mode) == set(['horizontalShift', 'SMILES']): 
            spec_list = []
            for spec in tqdm(spec_data, desc="spectra | {}".format(str(aug_mode))):
                spec_list += [torch.tensor(interpolate_spectrum(spec, new_x, orig_x)[0])] * (smi_aug_num + 1)# original spec
                
                shift = np.arange(-max_shift, max_shift + 1)
                shift = shift[shift != 0] # shift distance != 0
                assert aug_num <= 2*max_shift, "aug_num({}) should be less than 2*max_shift(2*{})".format(aug_num, max_shift)
                shift = np.random.choice(shift, size=aug_num, replace=False)
                
                for shift_i in shift:
                    spec_list += [self.shift_horizontal(spec, shift=shift_i, new_x=new_x, orig_x=orig_x)] * (smi_aug_num + 1)
            
        return spec_list
        
    
    def vertical_noise(self, spec_x, spec_y, orig_x, theta=0.01, alpha=1):
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

    def shift_horizontal(self, spectrum, shift, new_x, orig_x):
        shifted_x = orig_x + shift
        interp_func = interpolate.interp1d(shifted_x, spectrum, bounds_error=False, fill_value=0)
        shifted_spectrum = interp_func(new_x)
        return norm_spectrum(shifted_spectrum)

    def smiles2tensor(self, smi_data, vocab, aug_mode, aug_num, smi_aug_num=0):
        new_lines = []
        max_len = 0
        for smi in tqdm(smi_data, desc='smiles'):
            len_list = []

            smi_ori = torch.tensor([ vocab[token] for token in split_smiles(smi)])
            len_list.append(smi_ori.shape[0])
            if aug_mode is None: new_lines += [smi_ori]
            elif aug_mode == 'SMILES' or 'SMILES' in aug_mode:
                smi_tensor_list = [smi_ori]                

                sme = SmilesEnumerator()
                aug_smi_list = []
                j = 0
                smi_tryNum = {}
                while j < smi_aug_num:
                    smi_j = sme.randomize_smiles(smi)

                    # The number of some molecules' SMILES are less than aug num, like C1CCCC1, COC.
                    if smi_j in aug_smi_list: 
                        if smi_j in smi_tryNum: 
                            if smi_tryNum[smi_j] > smi_aug_num: pass
                            else: 
                                smi_tryNum[smi_j] += 1
                                continue
                        else: 
                            smi_tryNum[smi_j] = 1
                            continue
                    aug_smi_list.append(smi_j)

                    smi_j_ = torch.tensor([ vocab[token] for token in split_smiles(smi_j)])
                    len_list.append(smi_j_.shape[0])
                    smi_tensor_list.append(smi_j_)

                    j += 1
                
                if aug_mode == 'SMILES' or aug_mode == ['SMILES']:
                    new_lines += smi_tensor_list
                elif type(aug_mode) == list:
                    new_lines += smi_tensor_list * (aug_num+1)

            else:
                new_lines += [smi_ori] * (aug_num+1)
            if max(len_list) > max_len:
                max_len = max(len_list)
            
        return new_lines, max_len
    
    def formula2tensor(self, formula_data, vocab, aug_mode, aug_num, smi_aug_num):
        txt_split_list = [split_formula(i) for i in formula_data]
        new_lines = []
        max_len = 0
        for line_ in tqdm(txt_split_list, desc='formula'):
            line = torch.tensor([ vocab[token] for token in line_])
            
            if line.shape[0] > max_len:
                max_len = line.shape[0]
            

            if aug_mode is None: new_lines.append(line)
            elif aug_mode == 'SMILES' or aug_mode == ['SMILES']:
                    new_lines += [line] * (smi_aug_num+1)
            elif type(aug_mode) == list and 'SMILES' in aug_mode:
                new_lines += [line] * (aug_num+1) * (smi_aug_num+1)
            else: new_lines += [line] * (aug_num+1)
        return new_lines, max_len
    
    def __len__(self):
        assert len(self.spec_list) == len(self.smi_list), "The number of spectra data({}) does not equal to that of SMILES data({}).".format(len(self.spec_list),len(self.smi_list))
        if self.formula:
            assert len(self.spec_list) == len(self.formula_list), "The number of spectra data({}) does not equal to that of formula data({}).".format(len(self.spec_list), len(self.formula_list))
        return len(self.spec_list)

    def __getitem__(self, index):
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

