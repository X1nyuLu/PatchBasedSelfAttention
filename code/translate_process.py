import torch
import torchtext
import numpy as np

from utils.DatasetDataLoader import CreateDataloader, set_random_seed
from model.formula_spec_model import make_model as make_model_withFormula
from model.formula_spec_model import Batch as Batch_withFormula
from model.spec_model import make_model as make_model_onlySpec
from model.spec_model import Batch as Batch_onlySpec

from tqdm import tqdm
import logging
import time
import os


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
                 spec_embed, 
                 model=None, d_model=512, num_heads=8, 
                 layer_num=4, d_ff=2048, dropout=0.1,
                 dataloader=None, batchsize=128,
                 spec_mask_len=400, smiles_max_padding=40,
                 formula=False, vocab_formula=None, formula_max_padding=None,
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
        logger.info("Start time: {}".format(time.strftime("%X_%d_%b")))

        self.output_save = save_path
        
        self.spec_mask_len = spec_mask_len
        self.formula = formula
        self.formula_max_padding = formula_max_padding
        self.tgt_max_padding = smiles_max_padding
        self.batch_size = batchsize
        set_random_seed(seed)
        
        # Vocab
        if type(vocab_smiles) == str:
            assert os.path.exists(vocab_smiles), "vocab_smiles do not exist."
            self.tgt_vocab = torch.load(vocab_smiles)
        else:
            assert isinstance(vocab_smiles, torchtext.vocab.Vocab), "vocab_smiles should be None or str or torchtext.vocab.Vocab."
            self.tgt_vocab = vocab_smiles

        self.formula_vocab = None
        if formula:
            if type(vocab_formula) == str:
                assert os.path.exists(vocab_formula), "vocab_formula do not exist."
                self.formula_vocab = torch.load(vocab_formula)
            else:
                assert isinstance(vocab_formula, torchtext.vocab.Vocab), "vocab_formula should be None or str or torchtext.vocab.Vocab."
                self.formula_vocab = vocab_formula

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
        
        logger.info("")
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
                                                formula=formula,formula_max_pad=self.formula_max_padding)
            self.dataloader = createdataloader.dataloader(dataset, batch_size=self.batch_size, shuffle=True)
        else: self.dataloader = dataloader
        del dataset
        logger.info("test : {} batches per epoch".format(len(self.dataloader)))
        
        if type(model) == str:
            assert os.path.exists(model), "`model` do not exists."
            logger.info("Loading model: {}".format(model))
            assert spec_embed != None, "`spec_embed` is not set."

            if formula:
                self.model = make_model_withFormula(spec_embed=spec_embed, formula_vocab=len(self.formula_vocab), tgt_vocab=len(self.tgt_vocab),
                                                    d_model=d_model, h=num_heads, N=layer_num, d_ff=d_ff, dropout=dropout)
            else:
                self.model = make_model_onlySpec(tgt_vocab=len(self.tgt_vocab), src_embed=spec_embed,
                                                 d_model=d_model, h=num_heads, N=layer_num, d_ff=d_ff, dropout=dropout)

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
    
        
    def beam_search_optimized(self, formula, spec, beam_width, n_best):
        bs = spec.size(0)
        ys = torch.zeros(bs, 1).fill_(self.bos_idx).type_as(spec.data).long()

        if self.formula:
            batch = Batch_withFormula(self.device, formula, spec, ys, spec_mask_len=self.spec_mask_len, pad=self.padding_idx)
        else:
            batch = Batch_onlySpec(self.device, src=spec, tgt=ys, pad=self.padding_idx)

        with torch.no_grad():
            if self.formula:
                memory = self.model.encode(batch.formula, batch.spec, batch.src_mask)
                out = self.model.decode(
                        memory, ys, subsequent_mask(ys.size(1)).type_as(spec.data), batch.src_mask
                    )
            else:
                memory = self.model.encode(batch.src)
                out = self.model.decode(memory, ys, subsequent_mask(ys.size(1)).type_as(spec.data))
            
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
                    if self.formula:
                        out = self.model.decode(
                            memory, ys_new, subsequent_mask(ys_new.size(1)).type_as(spec.data), batch.src_mask
                            )
                    else:
                        out = self.model.decode(
                            memory, ys_new, subsequent_mask(ys_new.size(1)).type_as(spec.data)
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
            spec = batch["spec"].to(self.device)
            if self.formula:
                formula = batch["formula"].to(self.device)
            else: formula = None
            

            if mode == "G":
                pred = self.greedy_decode(formula, spec)
            elif mode == "B":
                pred = self.beam_search_optimized(formula, spec, beam_width, n_best)
            
            self.output_txtfile(pred,filename)
            self.output_txtfile(batch["smi"], "tgt.txt")
                
            if testInf and i==9: break
            torch.cuda.empty_cache()

    
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
      