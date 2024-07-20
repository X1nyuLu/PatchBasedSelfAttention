import torch
import torch.nn as nn

import utils.the_annotated_transformer as atf
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class FormulaSpecEmbed(nn.Module):
    def __init__(self, formula_vocab, spec_embed, d_model=512,
                #  spec_len=3200, patch_len=8, spec_vocab=100,
                 ):
        super(FormulaSpecEmbed, self).__init__()

        self.formula_embed = atf.Embeddings(d_model=d_model, vocab=formula_vocab)
        self.spec_embed = spec_embed

    def forward(self, formula, spec):
        formula_ = self.formula_embed(formula)
        spec_ = self.spec_embed(spec)

        return torch.cat((formula_, spec_), dim=1)

def make_model(
    spec_embed, formula_vocab, tgt_vocab, 
    N=4, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    
    c = copy.deepcopy
    attn = atf.MultiHeadedAttention(h, d_model)
    ff = atf.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = atf.PositionalEncoding(d_model, dropout)
    src_embed = FormulaSpecEmbed(formula_vocab, spec_embed, d_model)
    model = EncoderDecoder(
        atf.Encoder(atf.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        atf.Decoder(atf.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed,
        nn.Sequential(atf.Embeddings(d_model, tgt_vocab), c(position)),
        atf.Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.tgt_embed = tgt_embed
        self.decoder = decoder
        self.generator = generator

        self.src_position = atf.PositionalEncoding(d_model=512, dropout=0.1)

    def forward(self, formula, spec, src_mask, tgt, tgt_mask):
        """
        Take in and process masked src and target sequences.
        src_mask: formula mask + spectra mask
        """
        return self.decode(self.encode(formula, spec, src_mask), tgt, tgt_mask, src_mask)

    def encode(self, formula, spec, src_mask):
        se = self.src_embed(formula, spec)
        return self.encoder(self.src_position(se), src_mask)
        

    def decode(self, memory, tgt, tgt_mask, src_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

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