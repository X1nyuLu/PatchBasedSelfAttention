import os
from tqdm import tqdm
import logging
import time
import GPUtil


import torch
import torchtext
# from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# import sys
# sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/")
# from utils.DatasetDataLoader import CreateDataloader
# from modified_ViT.utils.DatasetDataLoader import generateDataset, CreateDataloader
# from modified_ViT.utils.DatasetDataLoader import CreateDataloader
# from utils.DatasetDataLoader import generateDataset, CreateDataloader, set_random_seed

# from model.formula_spec_model import make_model as make_model_withFormula
# from model.formula_spec_model import Batch as Batch_withFormula
# from model.spec_model import make_model as make_model_onlySpec
# from model.spec_model import Batch as Batch_onlySpec

import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")
import utils.the_annotated_transformer as atf
from DatasetDataLoader_smiles import CreateSMILESDataloader, set_random_seed
from train_smilesModel_process import Batch

from modified_ViT.model.formula_spec_model import *

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Translate_Transformer:
    def __init__(self, test_data, 
                 save_path, 
                 vocab_smiles,
                 model=None, d_model=512, num_heads=8, 
                 layer_num=4, d_ff=2048, dropout=0.1,
                 dataloader=None, batchsize=128,
                 src_max_pad=415, tgt_max_pad=40,
                 seed=0):
        
        """
        Parameters:
            vocab: 
                'str': path to formula_tgt_vocab.pt which contains src and tgt vocab 
             OR 'tuple': contains src and tgt vocab 
            model: 
                'str': path to model_state_dict
             OR 'model instance'  
            
        """
        logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(os.path.join(save_path,"translate.log"))])
        logger = logging.getLogger("INIT")

        self.output_save = save_path
        
        self.src_max_padding = src_max_pad
        self.tgt_max_padding = tgt_max_pad
        self.batch_size = batchsize
        set_random_seed(seed)
        # Vocab
        if type(vocab_smiles) == str:
            assert os.path.exists(vocab_smiles), "vocab_smiles do not exist."
            self.smi_vocab = torch.load(vocab_smiles)
        else:
            assert isinstance(vocab_smiles, torchtext.vocab.Vocab), "vocab_smiles should be None or str or torchtext.vocab.Vocab."
            self.smi_vocab = vocab_smiles

        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

        self.padding_idx = self.smi_vocab['<blank>']
        self.bos_idx = self.smi_vocab["<s>"]
        self.eos_idx = self.smi_vocab["</s>"]
        self.unk_idx = self.smi_vocab["<unk>"]
        # print(self.padding_idx)
        
        logger.info("Loading test dataset...")
        if type(test_data) == str and os.path.exists(test_data):
            logger.info("Data: {}".format(test_data))
            dataset = torch.load(test_data)
        else: 
            logger.info("Directly input dataset")
            dataset = test_data

        logger.info("Building dataloader...")
        if dataloader == None:
            createdataloader = CreateSMILESDataloader(pad_id=self.padding_idx,
                                                      bs_id=self.bos_idx,
                                                      eos_id=self.eos_idx,
                                                      src_max_padding=self.src_max_padding,
                                                      tgt_max_padding=self.tgt_max_padding)
            self.dataloader = createdataloader.dataloader(dataset, batch_size=self.batch_size, shuffle=True)
        else: self.dataloader = dataloader
        del dataset
        logger.info("test : {} batches per epoch".format(len(self.dataloader)))
        
        if type(model) == str:
            assert os.path.exists(model), "`model` do not exists."
            logger.info("Loading model: {}".format(model))
            self.model = atf.make_model(src_vocab=len(self.smi_vocab), tgt_vocab=len(self.smi_vocab),
                                    N=layer_num, d_model=d_model, d_ff=d_ff, dropout=dropout, h=num_heads)

            if '.pt' in model:
                    self.model.load_state_dict(torch.load(model))
            elif '.tar' in model:
                checkpoint = torch.load(model)
                self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.info("Loading model...")
            self.model = model
        self.model.to(self.device)
        self.model.eval()
    
    def greedy_decode(self, src):
        batch_size = src.size(0)
        ys = torch.zeros(batch_size, 1).fill_(self.bos_idx).to(torch.long).to(self.device)
        batch = Batch(self.device, src, ys, pad=self.padding_idx) 
             
        # batch = Batch(self.device, formula, spec, ys, pad=self.padding_idx, spec_mask_len=self.spec_mask_len)

        with torch.no_grad():
            # memory = self.model.encode(batch.formula, batch.spec, batch.src_mask)
            memory = self.model.encode(batch.src, batch.src_mask)
            
            for i in range(self.tgt_max_padding - ys.shape[-1]):
                out = self.model.decode(memory, batch.src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data),)
                prob = self.model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.unsqueeze(-1)
                ys = torch.cat([ys, next_word], dim=-1)

        return ys
    
    
        
    def beam_search_optimized(self,src, beam_width, n_best):
        bs = src.size(0)
        ys = torch.zeros(bs, 1).fill_(self.bos_idx).type_as(src.data).long()
        batch = Batch(self.device, src, ys, pad=self.padding_idx) 
        
        with torch.no_grad():
            memory = self.model.encode(batch.src, batch.src_mask)
            out = self.model.decode(memory, batch.src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
            
            # print(out.shape)
            # print(out)
            prob = self.model.generator(out[:, -1])
            vocab_size = prob.shape[-1]
            prob, next_chars = prob.topk(k=beam_width)

            # Initialize beams: [(start_symbol, prob_of_next_char, next_char) * beam_width]
            beams = [(ys, prob[:, i].reshape(bs, -1), next_chars[:, i].reshape(bs, -1)) for i in range(beam_width)]
            for j in range(self.tgt_max_padding - 2):
                next_beams = []
                all_next_prob = torch.empty(0)
                all_ys = torch.empty(0)
                all_next_char = torch.empty(0)

                for ys, prob_, char_ in beams:
                    ys_new = torch.cat([ys, char_], dim=1)
                    out = self.model.decode(memory, batch.src_mask, ys_new, subsequent_mask(ys_new.size(1)).type_as(src.data))
                    
                    next_prob = self.model.generator(out[:, -1])
                    prob = prob_.repeat(1, vocab_size) + next_prob
                    prob, next_chars = prob.topk(k=beam_width)

                    if all_next_prob.numel() == 0:
                        all_next_prob = prob
                        all_ys = ys_new.repeat(1, beam_width).reshape(bs, beam_width, -1)
                        all_next_char = next_chars
                    else: 
                        all_next_prob  = torch.cat([all_next_prob, prob], dim=1)
                        all_ys = torch.cat([all_ys, ys_new.repeat(1, beam_width).reshape(bs, beam_width, -1)], dim=1)
                        all_next_char = torch.cat([all_next_char, next_chars], dim=1)

                    next_beams += [(ys_new, prob[:, i].reshape(bs, -1), next_chars[:, i].reshape(bs, -1)) for i in range(beam_width)]
                kprob, kindex = all_next_prob.topk(k=beam_width)
                kchar = all_next_char.gather(1, kindex)
                kindex_ = kindex.unsqueeze(-1).expand(-1, -1, all_ys.size(-1))
                kys = all_ys.gather(1, kindex_)

                beams = [(kys[:, i, :], kprob[:, i].reshape(-1,1), kchar[:, i].reshape(-1,1)) for i in range(beam_width)]

            out = [beam[0] for beam in beams[:n_best]]
            out = torch.stack(out,dim=1)
            del beams, all_next_char, all_next_prob, all_ys
            
            return out.reshape(-1,out.shape[-1])
        
    def translate_main(self, mode="G", beam_width=None, n_best=None, testInf=False):
        """
        Para:
            mode: "G" or "B" 
                - "G": Greedy Search
                - "B": Beam Search
        """
        logger = logging.getLogger("Translate")
        if mode == "G": filename = "GreedySearch_result.txt"
        elif mode == "B": filename = "BeamSearch_BW{}_NB{}_result.txt".format(beam_width, n_best)

        logger.info("mode: {} | beam_width: {} | n_best: {}".format(mode, beam_width, n_best))
        for i, batch in tqdm(enumerate(self.dataloader), desc="Batch"):
            src = batch["src_smi"].to(self.device)

            if mode == "G":
                pred = self.greedy_decode(src)
            elif mode == "B":
                # pred = self.beam_search(src, beam_width, n_best)
                pred = self.beam_search_optimized(src, beam_width, n_best)
                pred = pred.view(-1, pred.size(-1))
            
            self.output_txtfile(pred, filename)
            # print(batch["smi"])
            self.output_txtfile(batch["tgt_smi"], "tgt.txt")
            # self.output_txtfile(batch["tgt"], "tgt.txt")
                
            if testInf and i==9: break
            torch.cuda.empty_cache()
            # del pred


    
    def output_txtfile(self, predictions, filename):
        """
        translate_results: torch.tensor
        """
        
        numpy_array = predictions.to("cpu").numpy()
        itos_dic = {index: value for index, value in enumerate(self.smi_vocab.get_itos())}
        vectorized_map = np.vectorize(itos_dic.get)
        mapped_array = vectorized_map(numpy_array)
        mapped_list = mapped_array.tolist()
        with open(os.path.join(self.output_save, filename), 'a') as file:
            for prediction in tqdm(mapped_list, desc="OutputFile"):
                prediction = prediction[1:] # remove <s>
                line = " ".join(prediction)
                line = line.split("</s>")[0] #remove </s>
                file.write(line + "\n")
                
    
    
if __name__ == "__main__":
    pass