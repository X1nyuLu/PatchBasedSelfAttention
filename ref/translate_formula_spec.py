import os
from tqdm import tqdm
import logging
import time
import GPUtil


import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/")
# from utils.DatasetDataLoader import CreateDataloader
# from modified_ViT.utils.DatasetDataLoader import generateDataset, CreateDataloader
from modified_ViT.utils.DatasetDataLoader import CreateDataloader


from modified_ViT.model.formula_spec_model import *

# import Transformer_Code.utils.generateDataset as gd
# import Transformer_Code.utils.createDataloader as dl

from AugmentData.train_formula_spec import Batch


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Translate_Transformer:
    def __init__(self, test_data, vocab,
                 model, patch_size,
                 d_model, num_heads, 
                 output_save, 
                 batch_size,
                 spec_mask_len=400,
                 dataloader=None,
                 spec_embed=None):
        
        """
        Parameters:
            vocab: 
                'str': path to formula_tgt_vocab.pt which contains src and tgt vocab 
             OR 'tuple': contains src and tgt vocab 
            model: 
                'str': path to model_state_dict
             OR 'model instance'  
            
        """
        logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(os.path.join(output_save,"translate.log"))])
        logger = logging.getLogger("INIT")

        self.output_save = output_save

        if type(vocab) == str and os.path.exists(vocab):
            self.formula_vocab, self.tgt_vocab = torch.load(vocab)
        else:
            self.formula_vocab, self.tgt_vocab = vocab

        self.patch_size = patch_size
        self.formula_max_padding = 15
        self.tgt_max_padding = 40
        self.spec_mask_len = spec_mask_len

        self.batch_size = batch_size

        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

        self.padding_idx = self.tgt_vocab['<blank>']
        self.bos_idx = self.tgt_vocab["<s>"]
        self.eos_idx = self.tgt_vocab["</s>"]
        self.unk_idx = self.tgt_vocab["<unk>"]
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
            createdataloader = CreateDataloader(pad_id=self.padding_idx,
                                                bs_id=self.bos_idx,
                                                eos_id=self.eos_idx,
                                                tgt_max_padding=self.tgt_max_padding,
                                                formula=True,formula_max_pad=self.formula_max_padding)
            self.dataloader = createdataloader.dataloader(dataset, batch_size=self.batch_size, shuffle=True)
        else: self.dataloader = dataloader
        del dataset
        logger.info("test : {} batches per epoch".format(len(self.dataloader)))
        
        if type(model) == str and os.path.exists(model):
            logger.info("Loading model: {}".format(model))
            assert spec_embed != None, "Spec_embed is not set."
            self.model = make_model(spec_embed=spec_embed, formula_vocab=len(self.formula_vocab), tgt_vocab=len(self.tgt_vocab),
                                    d_model=d_model, h=num_heads)
            # self.model.load_state_dict(torch.load(model))
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
    
    def greedy_decode(self, formula, spec,):
        batch_size = spec.size(0)
        ys = torch.zeros(batch_size, 1).fill_(self.bos_idx).to(torch.long).to(self.device)
        batch = Batch(self.device, formula, spec, ys, pad=self.padding_idx, spec_mask_len=self.spec_mask_len)

        with torch.no_grad():
            memory = self.model.encode(batch.formula, batch.spec, batch.src_mask)
            
            for i in range(self.tgt_max_padding - ys.shape[-1]):
                out = self.model.decode(
                    memory, ys, subsequent_mask(ys.size(1)).type_as(spec.data), batch.src_mask
                )
                prob = self.model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.unsqueeze(-1)
                ys = torch.cat([ys, next_word], dim=-1)

        return ys
    
    
        
    def beam_search_optimized(self, formula, spec, beam_width, n_best):
        bs = spec.size(0)
        ys = torch.zeros(bs, 1).fill_(self.bos_idx).type_as(spec.data).long()


        # batch = Batch(self.device, formula, spec, self.patch_size, ys, pad=self.padding_idx)
        batch = Batch(self.device, formula, spec, ys, spec_mask_len=self.spec_mask_len, pad=self.padding_idx)
        with torch.no_grad():
            memory = self.model.encode(batch.formula, batch.spec, batch.src_mask)
            out = self.model.decode(
                    memory, ys, subsequent_mask(ys.size(1)).type_as(spec.data), batch.src_mask
                )
            
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
                    out = self.model.decode(
                        memory, ys_new, subsequent_mask(ys_new.size(1)).type_as(spec.data), batch.src_mask
                        )
                    
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
                    # break
                kprob, kindex = all_next_prob.topk(k=beam_width)
                kchar = all_next_char.gather(1, kindex)
                kindex_ = kindex.unsqueeze(-1).expand(-1, -1, all_ys.size(-1))
                kys = all_ys.gather(1, kindex_)

                beams = [(kys[:, i, :], kprob[:, i].reshape(-1,1), kchar[:, i].reshape(-1,1)) for i in range(beam_width)]
                # print("beams:")
                # for i in beams: print(i)
                # print(beams)
            # out = [beam[0] for beam in beams[:n_best]]
            # out_ = []
            # for i in range(bs):
            #     out_ += [o[i] for o in out]
            # return [beam[0] for beam in beams[:n_best]]

            # print("beams: ")
            # print(beams)
            out = [beam[0] for beam in beams[:n_best]]
            # print(out)
            out = torch.stack(out,dim=1)
            # print(out.shape)
            # print(out.reshape(-1,out.shape[-1]))
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
            spec = batch["spec"].to(self.device)
            formula = batch["formula"].to(self.device)
            

            if mode == "G":
                pred = self.greedy_decode(formula, spec)
            elif mode == "B":
                # pred = self.beam_search(src, beam_width, n_best)
                pred = self.beam_search_optimized(formula, spec, beam_width, n_best)
                pred = pred.view(-1, pred.size(-1))
            
            self.output_txtfile(pred,filename)
            # print(batch["smi"])
            self.output_txtfile(batch["smi"], "tgt.txt")
            # self.output_txtfile(batch["tgt"], "tgt.txt")
                
            if testInf and i==10: break
            # if testInf and i==0: break


    
    def output_txtfile(self, predictions, filename):
        """
        translate_results: torch.tensor
        """
        
        numpy_array = predictions.to("cpu").numpy()
        itos_dic = {index: value for index, value in enumerate(self.tgt_vocab.get_itos())}
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
    
    # testdata_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/data/3208_data/test_set.pt"
    # testdata_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/formula_spec_train_result/embedpatchattention_withformula_14:34:06_22_Jan/data/test_set.pt"
    # testdata_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/formula_spec_train_result/embedpatchattention_withformula_all_10:54:52_23_Jan/data/test_set.pt"
    
    # testdata_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/newVocabNewData/train_result/embedpatchattention_withF_bs8_ps128_aughorizontalShift_10:56:16_29_Jan/data/test_set.pt"
    testdata_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/newVocabNewData/train_result/embedpatchattention_withF_bs8_ps128_augverticalNoise_10:58:08_29_Jan/data/test_set.pt"

    # data_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/"
    # formula_vocab, tgt_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/formula_tgt_vocab.pt")
    # dataset = generateDataset(os.path.join(data_path,"30data.pkl"), tgt_vocab, spec_len=3200, formula=True, formula_vocab=formula_vocab)
    # torch.save(dataset, os.path.join(data_path, 'dataset_30.pt'))
    # testdata_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/dataset_30.pt"

    # vocab = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/formula_tgt_vocab.pt"
    formula_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_formula.pt")
    smiles_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_smiles.pt")
    vocab = (formula_vocab, smiles_vocab)

    # save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/formula_spec_translate_result"
    save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/newVocabNewData/translate_result"
   
    patch_size = 8
    spec_len = 3200
    
    # EmbedPatchAttention
    spec_embed = EmbedPatchAttention(spec_len=spec_len, patch_len=patch_size, d_model=512, src_vocab=100)
    # model_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/formula_spec_train_result/embedpatchattention_withformula_all_10:54:52_23_Jan"
    # save_path = os.path.join(save_path,"embedpatchattention_beamsearch_{}".format(time.strftime("%X_%d_%b")))
    # model_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/newVocabNewData/train_result/embedpatchattention_withF_bs8_ps128_aughorizontalShift_10:56:16_29_Jan"
    # save_path = os.path.join(save_path,"embedpatchattention_augHorizontal_{}".format(time.strftime("%X_%d_%b")))
    model_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/newVocabNewData/train_result/embedpatchattention_withF_bs8_ps128_augverticalNoise_10:58:08_29_Jan"
    save_path = os.path.join(save_path,"embedpatchattention_augVertical_{}".format(time.strftime("%X_%d_%b")))

    # EmbedPatch 
    # spec_embed = EmbedPatch(spec_len=spec_len, patch_len=patch_size, d_model=512, src_vocab=100)
    # model_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/formula_spec_train_result/embedpatch_withformula_all_12:53:01_25_Jan"
    # save_path = os.path.join(save_path,"embedpatch_beamsearch_{}".format(time.strftime("%X_%d_%b")))

    model_path = os.path.join(model_dir, "best_model_params.pt")
    # model_path = os.path.join(model_dir, "EPOCH_40.pt")

    os.mkdir(save_path)
    print(save_path)
    # Translate = Translate_Transformer(dataset, vocab_path, spec_len=spec_len,
    Translate = Translate_Transformer(testdata_path, vocab, spec_embed=spec_embed,
                                      model=model_path, 
                                      d_model=512, num_heads=8,
                                      patch_size=patch_size,
                                      output_save=save_path,
                                      batch_size=512)
    Translate.translate_main(mode="B", beam_width=10, n_best=10, testInf=True)
    # Translate.translate_main(mode="G", testInf=True)
    

 