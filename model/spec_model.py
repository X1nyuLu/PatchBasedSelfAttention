import sys
sys.path.append("FinalResult")
import utils.the_annotated_transformer as atf
import torch.nn as nn
import copy
def make_model(
    tgt_vocab, src_embed, 
    N=4, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    
    c = copy.deepcopy
    attn = atf.MultiHeadedAttention(h, d_model)
    ff = atf.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = atf.PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        atf.Encoder(atf.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        atf.Decoder(atf.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(src_embed, c(position)),
        nn.Sequential(atf.Embeddings(d_model, tgt_vocab), c(position)),
        atf.Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
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

    def forward(self, src, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src), tgt, tgt_mask)

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, tgt, tgt_mask, src_mask=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, device, src, tgt, pad=2):  # 2 = <blank>
        
        self.src = src.to(device)
        # self.src_mask = (src != pad).unsqueeze(-2).to(device)
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