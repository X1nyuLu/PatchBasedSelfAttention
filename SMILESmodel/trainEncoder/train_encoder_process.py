import os
import time
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torchtext

import time


import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")

import utils.the_annotated_transformer as atf
from utils.DatasetDataLoader import generateDataset, CreateDataloader, set_random_seed
from model.spectra_process_layer import *
from model.formula_spec_model import make_model as make_model_withFormula
from model.formula_spec_model import Batch as Batch_withFormula
from model.spec_model import make_model as make_model_onlySpec
from model.spec_model import Batch as Batch_onlySpec
from SMILESmodel.trainSMILESmodel.train_smilesModel_process import Batch as SMILESBatch
# from SMILESmodel.DatasetDataLoader_smiles import CreateSMILESDataloader

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
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

class TrainVal:

    def __init__(self, data, save_path, 
                 vocab_smiles, 
                 spec_embed, spec_len=3200, spec_mask_len=400,
                 spec_model=None, smi_model=None,
                 d_model=512, num_heads=8, layer_num=4, d_ff=2048, dropout=0.1,
                 batchsize=128, accum_iter=10, num_epochs=200, warmup=3000, base_lr=1.0,
                 Resume=False,
                 aug_mode=None, testset_aug=None,
                 formula=False, vocab_formula=None, formula_max_padding=None,
                 smiles_max_padding=40,
                 testTrain=False, 
                 seed=0,
                 earlystop_delta=None,
                 earlystop_patience=None
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
                `model instance`
            aug_mode: None or 'verticalNoise' or 'horizontalShift'
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
        self.accum_iter= accum_iter
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

        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        
        set_random_seed(seed)
        
        logger.info("formula: {} | spec_len: {} | spec_mask_len: {} | d_model: {} | num_heads:{} ".format(self.formula, spec_len, self.spec_mask_len, self.d_model, self.num_heads))
        logger.info("tgt max pad: {} | batchsize: {}, accumulation iter: {} | epoch num: {}| base lr: {}| warmup stpes:{} | seed: {}".format(
                    self.tgt_max_padding, self.batch_size, self.accum_iter,  self.num_epochs, self.base_lr, self.warmup, seed))

        # CONSTANTS
        self.bs_idx = 0
        self.eos_idx = 1
        self.padding_idx = 2

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

        # Dataset
        logger.info("Building dataset...")
        logger.info("Data: {}".format(data))
        logger.info("Augmentation mode: {}".format(aug_mode))
        if os.path.isfile(data):
            assert os.path.exists(data) and '.pkl' in data, "`data` should be path to .pkl file."
            if vocab_smiles == None or vocab_formula == None:
                pass
            else:
                data_df = pd.read_pickle(data)
            # Split data before generating dataset in case test dataset 
            # contains the same molecules in train set during data augmentation.
            data_df = data_df.sample(frac=1) # shuffle dataframe
            data_num = data_df.shape[0]
            train_set = data_df[:int(data_num*0.7)]
            val_set = data_df[int(data_num*0.7):int(data_num*0.8)]
            test_set = data_df[int(data_num*0.8):]
            logger.info("train_set: {} data | val_set: {} data | test_set: {} data".format(train_set.shape[0],
                                                                                           val_set.shape[0],
                                                                                           test_set.shape[0]))

            data_path = os.path.join(self.save_path, 'data')
            if not os.path.exists(data_path): os.mkdir(data_path)

            train_set = generateDataset(train_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, aug_mode=aug_mode)
            torch.save(train_set, os.path.join(data_path, 'train_set.pt'))
            
            val_set = generateDataset(val_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, aug_mode=aug_mode)
            torch.save(val_set, os.path.join(data_path, 'val_set.pt'))
            
            test_set = generateDataset(test_set, smiles_vocab=self.tgt_vocab, spec_len=spec_len, formula=formula, formula_vocab=self.formula_vocab, aug_mode=testset_aug)
            torch.save(test_set, os.path.join(data_path, 'test_set.pt'))

            smi_max_len = max(train_set.smi_max_len, val_set.smi_max_len, test_set.smi_max_len)
            formula_max_len = None
            if formula: formula_max_len = max(train_set.formula_max_len, val_set.formula_max_len, test_set.formula_max_len)
            logger.info("smiles max len: {} | formula max len: {}".format(smi_max_len, formula_max_len))
        elif os.path.isdir(data):
            assert os.path.exists(data), "`data` do not exists."
            if testTrain:
                dataset = torch.load(os.path.join(data, 'val_set.pt'))
                train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
            else:
                train_set = torch.load(os.path.join(data, 'train_set.pt'))
                val_set = torch.load(os.path.join(data, 'val_set.pt'))
        else: train_set, val_set = data

        logger.info("Building dataloader...")
        createdataloader = CreateDataloader(pad_id=self.padding_idx,
                                            bs_id=self.bs_idx,
                                            eos_id=self.eos_idx,
                                            tgt_max_padding=self.tgt_max_padding,
                                            formula=formula, formula_max_pad=self.formula_max_padding)
        self.train_dataloader = createdataloader.dataloader(train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = createdataloader.dataloader(val_set, batch_size=self.batch_size, shuffle=True)
        logger.info("train : {} batches per epoch | val: {} batches per epoch".format(len(self.train_dataloader),len(self.valid_dataloader)))

        # Model
        logger.info("Load SMILES model: {}".format(smi_model))
        self.smi_model = atf.make_model(src_vocab=len(self.tgt_vocab), tgt_vocab=len(self.tgt_vocab),
                                        N=layer_num, d_model=d_model, d_ff=d_ff, dropout=dropout)
        checkpoint = torch.load(smi_model)
        self.smi_model.load_state_dict(checkpoint['model_state_dict'])
        for name, param in self.smi_model.named_parameters():
            param.requires_grad = False
        self.smi_model.to(self.device)
        self.smi_model.eval()

        logger.info("Building Spectra model...")
        assert spec_embed != None, "`spec_embed` is not set."
        if formula:
            self.model = make_model_withFormula(spec_embed=spec_embed, formula_vocab=len(self.formula_vocab), tgt_vocab=len(self.tgt_vocab),
                                                d_model=d_model, h=num_heads, N=layer_num, d_ff=d_ff, dropout=dropout)
        else:
            self.model = make_model_onlySpec(tgt_vocab=len(self.tgt_vocab), src_embed=spec_embed, N=layer_num, d_ff=d_ff, dropout=dropout)

        self.Resume = Resume
        if self.Resume:
            logger.info("Resume train model: {}".format(spec_model))
            checkpoint = torch.load(spec_model)
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
            if spec_model == None: pass
            elif type(spec_model)==str and os.path.isfile(spec_model):
                logger.info("Loading model: {}".format(spec_model))
                if '.pt' in spec_model:
                    self.model.load_state_dict(torch.load(spec_model))
                elif '.tar' in spec_model:
                    checkpoint = torch.load(spec_model)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model = spec_model

            self.model.to(self.device)
        for name, param in self.model.named_parameters():
            if ('decoder' in name) or ('tgt_embed' in name) or ('generator' in name):
                param.requires_grad = False
        logger.info(self.model)

    def train_worker(self):
        logger = logging.getLogger("TRAIN")
        
        model = self.model
        self.writer = SummaryWriter(log_dir=self.save_path)

        # criterion_ = nn.KLDivLoss(reduction="sum")
        # criterion = LabelSmoothing(
        #     criterion_, size=len(self.tgt_vocab), padding_idx=self.padding_idx, smoothing=0.1
        # )
        criterion = nn.SmoothL1Loss()
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
                                            # atf.SimpleLossCompute(model.generator, criterion),
                                            criterion,
                                            optimizer,
                                            lr_scheduler,
                                            mode="train",
                                            accum_iter=self.accum_iter,
                                            train_state=train_state,)
            

            if epoch % 20 == 0:
                file_path = os.path.join(model_savepath,"%s%.2d.pt" % (self.file_prefix, epoch))
                torch.save(model.state_dict(), file_path)
                torch.save(self.smi_model.state_dict(), os.path.join(model_savepath,"%s%.2d.pt" % ('smi_epoch', epoch)))
            torch.cuda.empty_cache()

            logger.info(f"Epoch {epoch} Validation ====")
            model.eval()
            val_loss, _ = self.run_epoch(epoch, self.valid_dataloader,
                                         model,
                                        #  atf.SimpleLossCompute(model.generator, criterion),
                                         criterion,
                                         atf.DummyOptimizer(),
                                         atf.DummyScheduler(),
                                         mode="eval",)
            self.writer.add_scalar("Val Loss", val_loss, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info('Epoch {} | Saving model with the best val loss: {}'.format(epoch, best_val_loss))
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state': lr_scheduler.state_dict()},
                            best_params_path)
                self.earlystop_step = 1
            elif val_loss > best_val_loss + self.earlystop_delta:
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
                  accum_iter=1,
                  train_state=TrainState(),
                ):
        """Train a single epoch"""
        logger = logging.getLogger("BATCH TRAIN")
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0

        src_size = self.spec_mask_len + self.formula_max_padding

        for i, b in enumerate(dataloader):
            if self.formula:
                batch = Batch_withFormula(self.device, b['formula'], b['spec'], b['smi'], spec_mask_len=self.spec_mask_len, pad=self.padding_idx) 
            else:
                batch = Batch_onlySpec(self.device, b['spec'], b['smi'], self.padding_idx) 
            
            with torch.no_grad():
                smi_ = pad(b['smi'], (0, src_size - self.tgt_max_padding), value=self.padding_idx)
                # print(b['smi'].shape, src_size, smi_.shape)
                smi_batch = SMILESBatch(self.device, smi_, pad=self.padding_idx) 
                smi_enc_out = self.smi_model.encode(smi_batch.src, smi_batch.src_mask)
                # print("smi enc out: ",smi_enc_out.shape)
            if mode == "train" :
                if self.formula:
                    enc_out = model.encode(
                        batch.formula, batch.spec, batch.src_mask
                    )
                    # print("enc out", enc_out.shape)
                else:
                    enc_out = model.encode(batch.src)
                
                # loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                loss_node = loss_compute(enc_out, smi_enc_out)
                loss = loss_node.data
                
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.tgt.shape[0]
                train_state.tokens += batch.ntokens
                if i % accum_iter == 0:
                    assert accum_iter > 0, "accum_iter should be larger than 0."
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                scheduler.step()
            elif mode == "eval": 
                with torch.no_grad():
                    if self.formula:
                        enc_out = model.encode(
                            batch.formula, batch.spec, batch.src_mask
                        )
                    else:
                        enc_out = model.encode(batch.src)
                    # loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                    loss_node = loss_compute(enc_out, smi_enc_out)
                    loss = loss_node.data

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 40 == 1 and mode == "train":
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                logger.info(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.3e "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss, tokens / elapsed, lr)
                    # % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                # self.writer.add_scalar("Train Loss",loss / batch.ntokens, i + epoch * len(self.train_dataloader))
                self.writer.add_scalar("Train Loss",loss, i + epoch * len(self.train_dataloader))
                start = time.time()
                tokens = 0

            del loss
            del loss_node
            torch.cuda.empty_cache()
        # return total_loss / total_tokens, train_state
        return total_loss/len(dataloader), train_state
    

# if __name__ == "__main__":
#     spec_embed = EmbedPatchAttention(spec_len=3200, patch_len=8, d_model=512, src_vocab=100)
#     model = make_model_withFormula(spec_embed=spec_embed, formula_vocab=30, tgt_vocab=30)
#     for name, param in model.named_parameters():
#         print(name)
                                    

#     data = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/IRtoMol/data/ir_data.pkl"
#     save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/newVocabNewData/train_result"
   
#     patch_size = 8
#     spec_len = 3200

#     # EmbedPatchAttention
#     batch_size = 128
#     spec_mask_len = int(spec_len/patch_size)
#     spec_embed = EmbedPatchAttention(spec_len=spec_len, patch_len=patch_size, d_model=512, src_vocab=100)
#     # augMode = None
#     augMode = 'verticalNoise'
#     # augMode = 'horizontalShift'
#     save_path = os.path.join(save_path,"embedpatchattention_withF_bs{}_ps{}_aug{}_{}".format(patch_size, batch_size, augMode,time.strftime("%X_%d_%b")))
    
    
    
#     os.mkdir(save_path)
    
#     formula_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_formula.pt")
#     smiles_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_smiles.pt")
#     vocab = (formula_vocab, smiles_vocab)

#     TRAIN = TrainVal(data, vocab, spec_embed, save_path, 
#                      batchsize=batch_size, accum_iter=20,
#                      d_model=512, num_heads=8, spec_mask_len=spec_mask_len,
#                      aug_mode=augMode,
#                      testTrain=False)
#     TRAIN.train_worker()
    # train_set = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/OnlySpec/test_workflow/data/test_set.pt"
    # formula=False
    
    # train_set = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/test_workflow/data/test_set.pt"
    # formula=True
    # createdataloader = CreateDataloader(pad_id=2, bs_id=0, eos_id=1, 
    #                                     tgt_max_padding=40,
    #                                     formula=formula, formula_max_pad=15)
    # train_dataloader = createdataloader.dataloader(torch.load(train_set), batch_size=10, shuffle=True)
    # for batch in train_dataloader:
    #     print(batch)
        

    