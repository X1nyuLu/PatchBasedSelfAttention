import os
import time
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/")
import Transformer_Code.train_annotated_transformer.the_annotated_transformer as atf
from modified_ViT.utils.newDatasetDataLoader_withAug import generateDataset, CreateDataloader
from modified_ViT.model.spectra_process_layer import *
from modified_ViT.model.formula_spec_model import make_model

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

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, device, formula, spec, tgt, spec_mask_len=400, pad=2):  # 2 = <blank>
        
        self.formula = formula.to(device)
        self.spec = spec.to(device)
        self.batch_size = spec.shape[0]

        self.formula_mask = (formula != pad).unsqueeze(-2).to(device)
        self.spec_mask = torch.ones(self.batch_size, 1, spec_mask_len, dtype=torch.bool).to(device)
        self.src_mask = torch.cat((self.formula_mask, self.spec_mask), dim=-1)

        tgt = tgt.to(device)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & atf.subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask   

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

class TrainVal:

    def __init__(self, data, vocab, spec_embed,
                 save_path,
                 model=None, d_model=512, num_heads=8, 
                 spec_mask_len=400,
                 batchsize=128, accum_iter=10,
                 testTrain=False, aug_mode=None, num_epochs=200,
                 Resume=False,
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
        

        self.save_path = save_path

        # Model Paras 
        # self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.formula_max_padding = 15
        self.spec_mask_len = spec_mask_len
        self.tgt_max_padding= 40

        # Train Paras
        self.batch_size = batchsize
        self.num_epochs = num_epochs
        self.accum_iter= accum_iter
        self.base_lr= 1.0
        self.warmup= 3000
        self.file_prefix= "EPOCH_"

        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        
        logger.info("tgt max pad: {}| batchsize: {}, accumulation iter: {}| d_model: {}, num_heads:{} | epoch num: {}| base lr: {}| warmup stpes:{}".format(
                    self.tgt_max_padding, self.batch_size, self.accum_iter, self.d_model, self.num_heads, self.num_epochs, self.base_lr, self.warmup))

        # CONSTANTS
        self.bs_idx = 0
        self.eos_idx = 1
        self.padding_idx = 2

        if type(vocab) == str and os.path.exists(vocab):
            self.formula_vocab, self.tgt_vocab = torch.load(vocab)
        else:
            self.formula_vocab, self.tgt_vocab = vocab
        # self.formula_vocab, self.tgt_vocab = torch.load(vocab)

        logger.info("Building dataset...")
        logger.info("Data: {}".format(data))
        logger.info("Augmentation mode: {}".format(aug_mode))
        if os.path.isfile(data):
            dataset = generateDataset(data, self.tgt_vocab, spec_len=spec_len, formula=True, formula_vocab=self.formula_vocab, aug_mode=augMode)
            train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))
            data_path = os.path.join(self.save_path, 'data')
            os.mkdir(data_path)
            torch.save(train_set, os.path.join(data_path, 'train_set.pt'))
            torch.save(val_set, os.path.join(data_path, 'val_set.pt'))
            torch.save(test_set, os.path.join(data_path, 'test_set.pt'))
            logger.info("smiles max len: {}".format(dataset.smi_max_len))
        elif os.path.isdir(data):
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
                                            formula=True, formula_max_pad=self.formula_max_padding)
        self.train_dataloader = createdataloader.dataloader(train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = createdataloader.dataloader(val_set, batch_size=self.batch_size, shuffle=True)

        logger.info("train : {} batches per epoch | val: {} batches per epoch".format(len(self.train_dataloader),len(self.valid_dataloader)))
        self.Resume = Resume
        if self.Resume:
            logger.info("Resume train model: {}".format(model))
            checkpoint = torch.load(model)
            self.model = make_model(spec_embed=spec_embed, formula_vocab=len(self.formula_vocab), tgt_vocab=len(self.tgt_vocab),
                                        # spec_len=spec_len, patch_len=patch_size, 
                                        d_model=d_model, h=num_heads)
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
            if model == None:
                logger.info("Building model...")
                assert spec_embed != None, "Spec_embed is not set."
                self.model = make_model(spec_embed=spec_embed, formula_vocab=len(self.formula_vocab), tgt_vocab=len(self.tgt_vocab),
                                        d_model=d_model, h=num_heads)
            elif type(model)==str and os.path.isfile(model):
                logger.info("Loading model: {}".format(model))
                assert spec_embed != None, "Spec_embed is not set."
                self.model = make_model(spec_embed=spec_embed, formula_vocab=len(self.formula_vocab), tgt_vocab=len(self.tgt_vocab),
                                        d_model=d_model, h=num_heads)
                if '.pt' in model:
                    self.model.load_state_dict(torch.load(model))
                elif '.tar' in model:
                    checkpoint = torch.load(model)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model = model

            self.model.to(self.device)
        logger.info(self.model)

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

        best_val_loss = float('inf')
        # best_model_params_path = os.path.join(self.save_path, "best_model_params.pt")
        best_params_path = os.path.join(self.save_path, "best_model_optimizer_params.tar")
        
        for epoch in range(self.num_epochs):

            logger.info(f"Epoch {epoch} Training ====")
            model.train()
            _, train_state = self.run_epoch(epoch,
                                            (Batch(self.device, b['formula'], b['spec'], b['smi'],  
                                                   spec_mask_len=self.spec_mask_len, pad=self.padding_idx) for b in self.train_dataloader),
                                            model,
                                            atf.SimpleLossCompute(model.generator, criterion),
                                            optimizer,
                                            lr_scheduler,
                                            mode="train",
                                            accum_iter=self.accum_iter,
                                            train_state=train_state,)
            

            if epoch % 20 == 0:
                file_path = os.path.join(self.save_path,"%s%.2d.pt" % (self.file_prefix, epoch))
                torch.save(model.state_dict(), file_path)
            torch.cuda.empty_cache()

            logger.info(f"Epoch {epoch} Validation ====")
            model.eval()
            val_loss, _ = self.run_epoch(epoch,
                                      (Batch(self.device, b['formula'], b['spec'], b['smi'], 
                                             spec_mask_len=self.spec_mask_len, pad=self.padding_idx) for b in self.valid_dataloader),
                                      model,
                                      atf.SimpleLossCompute(model.generator, criterion),
                                      atf.DummyOptimizer(),
                                      atf.DummyScheduler(),
                                      mode="eval",)
            self.writer.add_scalar("Val Loss", val_loss, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info('Epoch {}: Saving model with the best val loss...'.format(epoch))
                # torch.save(model.state_dict(), best_model_params_path)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state': lr_scheduler.state_dict()},
                            best_params_path)     
                
            torch.cuda.empty_cache()

        self.writer.close()
        file_path = os.path.join(self.save_path, "%sfinal.pt" % self.file_prefix)
        torch.save(model.state_dict(), file_path)

        
    
    def run_epoch(self,
                  epoch,
                  data_iter,
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
        for i, batch in enumerate(data_iter):
            
            if mode == "train" :
                out = model.forward(
                    batch.formula, batch.spec, batch.src_mask, batch.tgt, batch.tgt_mask
                )
                loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.tgt.shape[0]
                train_state.tokens += batch.ntokens
                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                scheduler.step()
            elif mode == "eval": 
                with torch.no_grad():
                    out = model.forward(
                        batch.formula, batch.spec, batch.src_mask, batch.tgt, batch.tgt_mask
                    )
                    loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            

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
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                self.writer.add_scalar("Train Loss",loss / batch.ntokens, i + epoch * len(self.train_dataloader))
                start = time.time()
                tokens = 0

            del loss
            del loss_node
            torch.cuda.empty_cache()
        return total_loss / total_tokens, train_state
    

if __name__ == "__main__":

    data = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/IRtoMol/data/ir_data.pkl"
    save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/newVocabNewData/train_result"
   
    patch_size = 8
    spec_len = 3200

    # EmbedPatchAttention
    batch_size = 128
    spec_mask_len = int(spec_len/patch_size)
    spec_embed = EmbedPatchAttention(spec_len=spec_len, patch_len=patch_size, d_model=512, src_vocab=100)
    # augMode = None
    augMode = 'verticalNoise'
    # augMode = 'horizontalShift'
    save_path = os.path.join(save_path,"embedpatchattention_withF_bs{}_ps{}_aug{}_{}".format(patch_size, batch_size, augMode,time.strftime("%X_%d_%b")))
    
    
    
    os.mkdir(save_path)
    
    formula_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_formula.pt")
    smiles_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/utils/buildVocab/vocab_smiles.pt")
    vocab = (formula_vocab, smiles_vocab)

    TRAIN = TrainVal(data, vocab, spec_embed, save_path, 
                     batchsize=batch_size, accum_iter=20,
                     d_model=512, num_heads=8, spec_mask_len=spec_mask_len,
                     aug_mode=augMode,
                     testTrain=False)
    TRAIN.train_worker()

    