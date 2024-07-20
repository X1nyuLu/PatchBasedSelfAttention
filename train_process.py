import os
import time
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torchtext

import time
import pandas as pd

# import sys
# sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/")
# import Transformer_Code.train_annotated_transformer.the_annotated_transformer as atf
# from modified_ViT.utils.newDatasetDataLoader_withAug import generateDataset, CreateDataloader
# from modified_ViT.model.spectra_process_layer import *
# from modified_ViT.model.formula_spec_model import make_model

import utils.the_annotated_transformer as atf
from utils.DatasetDataLoader import generateDataset, CreateDataloader, set_random_seed
from model.spectra_process_layer import *

from model.formula_spec_model import make_model as make_model_withFormula
from model.formula_spec_model import Batch as Batch_withFormula
from model.spec_model import make_model as make_model_onlySpec
from model.spec_model import Batch as Batch_onlySpec

from buildVocab.vocab import build_vocab

from torch.utils.tensorboard import SummaryWriter

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, criterion, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # self.criterion = nn.KLDivLoss(reduction="sum")
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class TrainState:
    """Track number of steps, examples, and tokens processed"""
    
    step: int = 0  # Steps in the current epoch
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed
    

class TrainVal:

    def __init__(self, data, save_path, 
                 vocab_smiles, 
                 spec_embed, spec_len=3200, spec_mask_len=400,
                 model=None, d_model=512, num_heads=8, layer_num=4, d_ff=2048, dropout=0.1,
                 batchsize=128, num_epochs=200, warmup=3000, base_lr=1.0, 
                 Resume=False, dataset_mode=None,
                 aug_mode=None, testset_aug=None, aug_num=0, smi_aug_num=0,
                 max_shift=None, theta=None, alpha=None,
                 formula=False, vocab_formula=None, formula_max_padding=None,
                 smiles_max_padding=40,
                 split_testdata=True,
                 testTrain=False, 
                 seed=0,
                 earlystop_delta=None,
                 earlystop_patience=None,
                 report_step=40,
                 check_point_step=20
                 ):
        
        """
        Parameters:
            data: 
                `str: path to .pkl file` contains smiles, formula and spectra
                `str: path to dir` contains train/val/test dataset
            vocab: 
                `str: path to formula_tgt_vocab.pt` or `tuple` or `None`
                `str`: load vocab
                `tuple`: (formula_vocab, tgt_vocab)
                `None`: build vocab
            save_path: 
                saving output including vocab, tensorboard, log file etc.
            model: 
                `None`: build a new model
                ``str: path to model.pt / .tar
                `nn.Module` instance
            aug_mode: None or 'verticalNoise' or 'horizontalShift' or 'horizontalShiftNonFP'
        """
        logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(os.path.join(save_path,"train.log"))])
        logger = logging.getLogger("INIT")
        logger.info("Start time: {}".format(time.strftime("%X_%d_%b")))

        self.save_path = save_path
        self.formula = formula

        # Model Paras 
        self.d_model = d_model
        self.num_heads = num_heads
        self.spec_mask_len = spec_mask_len
        self.tgt_max_padding= smiles_max_padding
        self.formula_max_padding = formula_max_padding

        # Train Paras
        self.batch_size = batchsize
        self.num_epochs = num_epochs
        self.base_lr= base_lr
        self.warmup= warmup
        self.file_prefix= "EPOCH_"
        self.earlystop_delta = earlystop_delta
        self.earlystop_patience = earlystop_patience
        if self.earlystop_delta != None and self.earlystop_patience != None:
            logger.info("Early stop is set up: patience: {} | delta: {}".format(self.earlystop_patience, self.earlystop_delta))
            self.earlystop = True
        else:
            logger.info('Early stop is not set up.')
            self.earlystop = False
        
        self.report_step = report_step
        self.check_point_step = check_point_step


        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        # print(torch.cuda.is_available(), self.device)
        
        set_random_seed(seed)
        
        logger.info("spec_len: {} | spec_mask_len: {} | formula: {} | d_model: {} | num_heads:{} ".format(
                     spec_len, self.spec_mask_len, self.formula, self.d_model, self.num_heads))
        logger.info("formula max pad: {} | tgt max pad: {} | batchsize: {} | epoch num: {}| base lr: {}| warmup stpes:{} | seed: {}".format(
                    self.formula_max_padding, self.tgt_max_padding, self.batch_size, self.num_epochs, self.base_lr, self.warmup, seed))

        # CONSTANTS
        self.bs_idx = 0
        self.eos_idx = 1
        self.padding_idx = 2

        data_df = None

        # Vocab
        if vocab_smiles == None:
            assert os.path.exists(data) and '.pkl' in data, "`data` should be path to .pkl file."
            data_df = pd.read_pickle(data)
            vocab_path = os.path.join(save_path,'vocab')
            if not os.path.exists(vocab_path): os.mkdir(vocab_path)
            self.tgt_vocab = build_vocab(data_df['smiles'], vocab_path, 'vocab_smiles', mode='smiles')
        elif type(vocab_smiles) == str:
            assert os.path.exists(vocab_smiles), "vocab_smiles do not exist."
            self.tgt_vocab = torch.load(vocab_smiles)
        else:
            assert isinstance(vocab_smiles, torchtext.vocab.Vocab), "vocab_smiles should be None or str or torchtext.vocab.Vocab."
            self.tgt_vocab = vocab_smiles

        self.formula_vocab = None
        if formula:
            if vocab_formula == None:
                if vocab_smiles != None:
                    assert os.path.exists(data) and '.pkl' in data, "`data` should be path to .pkl file."
                    data_df = pd.read_pickle(data)
                    vocab_path = os.path.join(save_path,'vocab')
                    if not os.path.exists(vocab_path): os.mkdir(vocab_path)
                self.formula_vocab = build_vocab(data_df['formula'], vocab_path, 'vocab_formula', mode='formula')
            elif type(vocab_formula) == str:
                assert os.path.exists(vocab_formula), "vocab_formula do not exist."
                self.formula_vocab = torch.load(vocab_formula)
            else:
                assert isinstance(vocab_formula, torchtext.vocab.Vocab), "vocab_formula should be None or str or torchtext.vocab.Vocab."
                self.formula_vocab = vocab_formula

        # Dataset and Dataloader
        self.buildDatasetDataloader(data, data_df, spec_len, formula, logger,# vocab_smiles, vocab_formula, 
                                    dataset_mode,
                                    aug_mode, testset_aug, aug_num, smi_aug_num,
                                    max_shift, theta, alpha, split_testdata, testTrain)
        
        

        # Model
        if not isinstance(model, nn.Module):
            logger.info("Building model...")
            assert spec_embed != None, "`spec_embed` is not set."
            if formula:
                self.model = make_model_withFormula(spec_embed=spec_embed, formula_vocab=len(self.formula_vocab), tgt_vocab=len(self.tgt_vocab),
                                                    d_model=d_model, h=num_heads, N=layer_num, d_ff=d_ff, dropout=dropout)
            else:
                self.model = make_model_onlySpec(tgt_vocab=len(self.tgt_vocab), src_embed=spec_embed, N=layer_num, d_ff=d_ff, dropout=dropout)

        self.Resume = Resume
        if self.Resume:
            logger.info("Resume train model: {}".format(model))
            checkpoint = torch.load(model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.base_lr, betas=(0.9, 0.98), eps=1e-9
            )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler = LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda step: atf.rate(
                    step, self.d_model, factor=1, warmup=self.warmup
                ),
            )
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        else:
            if model == None: pass
            elif type(model)==str and os.path.isfile(model):
                logger.info("Loading model: {}".format(model))
                if '.pt' in model:
                    self.model.load_state_dict(torch.load(model))
                elif '.tar' in model:
                    checkpoint = torch.load(model)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(model, nn.Module):
                logger.info("Successful load model.")
                self.model = model

            self.model.to(self.device)
        logger.info(self.model)
    
    def buildDatasetDataloader(self, data, data_df, spec_len, formula, logger, #vocab_smiles, vocab_formula, 
                               dataset_mode=None,
                               aug_mode=None, testset_aug=None, aug_num=0, smi_aug_num=0,
                               max_shift=None, theta=None, alpha=None,
                               split_testdata=True, testTrain=False):
        logger.info("Building dataset...")
        logger.info("Data: {}".format(data))
        logger.info("Augmentation mode: {}".format(aug_mode))
        if os.path.isfile(data):
            assert os.path.exists(data) and '.pkl' in data, "`data` should be path to .pkl file."
            if data_df is None:
                data_df = pd.read_pickle(data)

            # Split data before generating dataset in case test dataset 
            # contains the same molecules in train set during data augmentation.
            data_df = data_df.sample(frac=1) # shuffle dataframe
            data_num = data_df.shape[0]

            data_path = os.path.join(self.save_path, 'data')
            if not os.path.exists(data_path): os.mkdir(data_path)

            if split_testdata:
                # train_set = data_df[:int(data_num*0.7)]
                # val_set = data_df[int(data_num*0.7):int(data_num*0.8)]
                # test_set = data_df[int(data_num*0.8):]

                train_set = data_df[:int(data_num*0.85)]
                val_set = data_df[int(data_num*0.85):int(data_num*0.9)]
                test_set = data_df[int(data_num*0.9):]
                logger.info("Original | train_set: {} data | val_set: {} data | test_set: {} data".format(train_set.shape[0],
                                                                                            val_set.shape[0],
                                                                                            test_set.shape[0]))
                
                if testset_aug != None:
                    test_set_ = generateDataset(test_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, 
                                                aug_mode=None, dataset_mode=dataset_mode)
                                                # smiles_max_pad=self.tgt_max_padding, formula_max_pad=self.formula_max_padding)
                    torch.save(test_set_, os.path.join(data_path, 'test_set_noAug.pt'))
                    del test_set_

                test_set = generateDataset(test_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, 
                                           aug_mode=testset_aug, aug_num=aug_num, max_shift=max_shift, theta=theta, alpha=alpha,
                                           smi_aug_num=smi_aug_num, dataset_mode=dataset_mode)
                                        #    smiles_max_pad=self.tgt_max_padding, formula_max_pad=self.formula_max_padding)
                torch.save(test_set, os.path.join(data_path, 'test_set.pt'))
                test_data_num = len(test_set)
                test_smi_max_len = test_set.smi_max_len
                if formula: test_f_max_len =  test_set.formula_max_len
                del test_set
            else:
                # train_set = data_df[:int(data_num*0.9)]
                # val_set = data_df[int(data_num*0.9):]

                train_set = data_df[:int(data_num*0.95)]
                val_set = data_df[int(data_num*0.95):]
                logger.info("Original | train_set: {} data | val_set: {} data".format(train_set.shape[0],
                                                                                      val_set.shape[0]))
                
            

            train_set = generateDataset(train_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, 
                                        aug_mode=aug_mode, aug_num=aug_num, max_shift=max_shift, theta=theta, alpha=alpha,
                                        smi_aug_num=smi_aug_num, dataset_mode=dataset_mode)
                                        # smiles_max_pad=self.tgt_max_padding, formula_max_pad=self.formula_max_padding)
            torch.save(train_set, os.path.join(data_path, 'train_set.pt'))
            
            # val_set = generateDataset(val_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, 
            #                           aug_mode=aug_mode, aug_num=aug_num, max_shift=max_shift, theta=theta, alpha=alpha,
            #                           smi_aug_num=smi_aug_num, dataset_mode=dataset_mode)

            val_set = generateDataset(val_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, 
                                                aug_mode=None, dataset_mode=dataset_mode)
            
            torch.save(val_set, os.path.join(data_path, 'val_set.pt'))
            
            
            if aug_mode != None:
                logger.info("AugMode: {} | AugNum: {} | MaxShift: {} | theta:{} | alpha:{} ".format(
                            str(aug_mode), aug_num, max_shift, theta, alpha))
                if split_testdata:
                    logger.info("After augmentation| train_set: {} data | val_set: {} data | test_set: {} data".format(len(train_set), len(val_set), test_data_num))
                else: logger.info("After augmentation| train_set: {} data | val_set: {} data".format(len(train_set), len(val_set)))

            formula_max_len = None
            if split_testdata:    
                smi_max_len = max(train_set.smi_max_len, val_set.smi_max_len, test_smi_max_len)
                if formula: formula_max_len = max(train_set.formula_max_len, val_set.formula_max_len, test_f_max_len)
            else:
                smi_max_len = max(train_set.smi_max_len, val_set.smi_max_len)
                if formula: formula_max_len = max(train_set.formula_max_len, val_set.formula_max_len)
            logger.info("smiles max len: {} | formula max len: {}".format(smi_max_len, formula_max_len))
            assert smi_max_len <= self.tgt_max_padding, "SMILES max length ({}) is larger than SMILES max padding ({}).".format(smi_max_len, self.tgt_max_padding)
            if formula: assert formula_max_len <= self.formula_max_padding, "Formula max length ({}) is larger than formula max padding ({}).".format(formula_max_len, self.formula_max_padding)
        elif os.path.isdir(data):
            assert os.path.exists(data), "`data` do not exists."
            if testTrain:
                dataset = torch.load(os.path.join(data, 'val_set.pt'))
                train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
            else:
                train_set = torch.load(os.path.join(data, 'train_set.pt'))
                val_set = torch.load(os.path.join(data, 'val_set.pt'))
            logger.info("train_set: {} data | val_set: {} data".format(len(train_set), len(val_set)))
                
        else: 
            train_set, val_set = data
            logger.info("train_set: {} data | val_set: {} data".format(len(train_set), len(val_set)))
            

        logger.info("Building dataloader...")
        createdataloader = CreateDataloader(pad_id=self.padding_idx,
                                            bs_id=self.bs_idx,
                                            eos_id=self.eos_idx,
                                            tgt_max_padding=self.tgt_max_padding,
                                            formula=formula, formula_max_pad=self.formula_max_padding)
        self.train_dataloader = createdataloader.dataloader(train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = createdataloader.dataloader(val_set, batch_size=self.batch_size, shuffle=True)
        logger.info("train : {} batches per epoch | val: {} batches per epoch".format(len(self.train_dataloader),len(self.valid_dataloader)))
        del train_set, val_set
        
    def train_worker(self):
        logger = logging.getLogger("TRAIN")
        
        model = self.model
        
        self.writer = SummaryWriter(log_dir=self.save_path)

        criterion_ = nn.KLDivLoss(reduction="sum")
        criterion = LabelSmoothing(
            criterion_, size=len(self.tgt_vocab), padding_idx=self.padding_idx, smoothing=0.1
        )
        criterion.to(self.device)

        if self.Resume:
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler
        else:
        
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.base_lr, betas=(0.9, 0.98), eps=1e-9
            )
            lr_scheduler = LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda step: atf.rate(
                    step, self.d_model, factor=1, warmup=self.warmup
                ),
            )
        train_state = TrainState()

        model_savepath = os.path.join(self.save_path, 'model')
        if not os.path.exists(model_savepath): os.mkdir(model_savepath)
        best_val_loss = float('inf')
        best_params_path = os.path.join(model_savepath, "best_model_optimizer_params.tar")
        if self.earlystop:
            self.earlystop_step = 1
        
        for epoch in range(self.num_epochs):

            logger.info(f"Epoch {epoch} Training ====")
            model.train()
            _, train_state = self.run_epoch(epoch, self.train_dataloader,
                                            model,
                                            atf.SimpleLossCompute(model.generator, criterion),
                                            optimizer,
                                            lr_scheduler,
                                            mode="train",
                                            train_state=train_state,
                                            report_step=self.report_step)
            

            if epoch % self.check_point_step == 0:
                file_path = os.path.join(model_savepath,"%s%.2d.pt" % (self.file_prefix, epoch))
                torch.save(model.state_dict(), file_path)
            torch.cuda.empty_cache()

            logger.info(f"Epoch {epoch} Validation ====")
            model.eval()
            val_loss, _ = self.run_epoch(epoch, self.valid_dataloader,
                                         model,
                                         atf.SimpleLossCompute(model.generator, criterion),
                                         atf.DummyOptimizer(),
                                         atf.DummyScheduler(),
                                         mode="eval",
                                         report_step=self.report_step)
            self.writer.add_scalar("Val Loss", val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info('Epoch {} | Saving model with the best val loss: {}'.format(epoch, best_val_loss))
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state': lr_scheduler.state_dict()},
                            best_params_path)
                if self.earlystop: self.earlystop_step = 1     
                
            if self.earlystop and (val_loss > best_val_loss + self.earlystop_delta):
                self.earlystop_step += 1
                if self.earlystop_step > self.earlystop_patience: 
                    logger.info("Early stop at epoch {}. val_loss: {} | best_val_loss: {}".format(epoch, val_loss, best_val_loss))
                    break     
                
            torch.cuda.empty_cache()

        self.writer.close()
        file_path = os.path.join(model_savepath, "%sfinal.pt" % self.file_prefix)
        torch.save(model.state_dict(), file_path)
    
    def run_epoch(self,
                  epoch,
                  dataloader, #data_iter,
                  model,
                  loss_compute,
                  optimizer,
                  scheduler,
                  mode="train",
                  train_state=TrainState(),
                  report_step=40,
                ):
        """Train a single epoch"""
        logger = logging.getLogger("BATCH TRAIN")
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0

        if self.formula: data_iter = (Batch_withFormula(
             self.device, b['formula'], b['spec'], b['smi'], spec_mask_len=self.spec_mask_len, pad=self.padding_idx) 
             for b in dataloader
        )
        else: data_iter = (Batch_onlySpec(
            self.device, b['spec'], b['smi'], self.padding_idx) 
            for b in dataloader
        )
        
        for i, batch in enumerate(data_iter):
            
            if mode == "train" :
                if self.formula:
                    out = model.forward(
                        batch.formula, batch.spec, batch.src_mask, batch.tgt, batch.tgt_mask
                    )
                else:
                    out = model.forward(
                        batch.src, batch.tgt, batch.tgt_mask
                    )
                loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.tgt.shape[0]
                train_state.tokens += batch.ntokens
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            elif mode == "eval": 
                with torch.no_grad():
                    if self.formula:
                        out = model.forward(
                            batch.formula, batch.spec, batch.src_mask, batch.tgt, batch.tgt_mask
                        )
                    else:
                        out = model.forward(
                            batch.src, batch.tgt, batch.tgt_mask
                        )
                    loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % report_step == 1 and mode == "train":
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                logger.info(
                    (
                        "Epoch Step: %6d | Loss: %6.3e "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.3e"
                    )
                    % (i, loss / batch.ntokens, tokens / elapsed, lr)
                )
                self.writer.add_scalar("Train Loss",loss / batch.ntokens, i + epoch * len(self.train_dataloader))
                start = time.time()
                tokens = 0

            del loss
            del loss_node
            torch.cuda.empty_cache()
        return total_loss / total_tokens, train_state
    
    