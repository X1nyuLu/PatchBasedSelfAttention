import torch
import torch.nn as nn
from torch.nn.functional import pad


import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/")
# import Transformer_Code.train_annotated_transformer.the_annotated_transformer as atf
import FinalResult.code.utils.the_annotated_transformer as atf
# import IRtoMol.scripts.prepare_data as prepdata
# import Transformer_Code.train_main_singleGPU as tms
# from modified_ViT.model.ref.PatchTST_layer import *

# import pandas as pd
# import numpy as np

class MaxPool(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        """
        src_vocab: from 0 to 99, 100 values in total.
        """
        super(MaxPool, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.maxpool = nn.MaxPool1d(kernel_size=patch_len, stride=patch_len)
        self.embed = atf.Embeddings(d_model, src_vocab)
        
    def forward(self, spec):
        return self.embed(self.maxpool(spec).to(torch.int)).squeeze(1)

class AvgPool(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(AvgPool, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.avgpool = nn.AvgPool1d(kernel_size=patch_len, stride=patch_len)
        self.embed = atf.Embeddings(d_model, src_vocab)
        
    def forward(self, spec):
        return self.embed(self.avgpool(spec).to(torch.int)).squeeze(1)

class EmbedPatch(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(EmbedPatch, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        
    def forward(self, spec): #[batch_size, 1, spec_len]
        batch_size = spec.shape[0]
        spec = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]
        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(batch_size, spec.shape[1], -1) #[batch_size, spec_len/patch_size, patch_size*d_model]
        return self.patch(spec)

class EmbedPatchAttention(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(EmbedPatchAttention, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = atf.MultiHeadedAttention(h=8, d_model=512)
        
    def forward(self, spec): #[batch_size, 1, spec_len]
        batch_size = spec.shape[0]
        spec = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(-1, self.patch_len, self.d_model) #[batch_size*(spec_len/patch_size), patch_size, d_model]
        spec = self.attention(spec, spec, spec)

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(batch_size, spec.shape[1], -1)
        
        return self.patch(spec)


class EmbedPatchDiffAttention(nn.Module):
    def __init__(self, d_model=512, src_vocab=100) -> None:
        super(EmbedPatchDiffAttention, self).__init__()
        """
        spectra length: 3200
        """
        # assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        # self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)

        self.attention1 = atf.MultiHeadedAttention(h=8, d_model=512)
        self.patch1 = nn.Linear(16*d_model, d_model)

        self.attention2 = atf.MultiHeadedAttention(h=8, d_model=512)
        self.patch2 = nn.Linear(32*d_model, d_model)

        self.attention3 = atf.MultiHeadedAttention(h=8, d_model=512)
        self.patch3 = nn.Linear(64*d_model, d_model)

        self.attention4 = atf.MultiHeadedAttention(h=8, d_model=512)
        self.patch4 = nn.Linear(128*d_model, d_model)
        
    def forward(self, spec): #[batch_size, 1, spec_len]
        batch_size = spec.shape[0]
        spec_embed = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec_16 = spec_embed.view(batch_size, -1, 16, self.d_model) #[batch_size, spec_len/16=200, 16, d_model]
        spec_16 = spec_16.view(-1, 16, self.d_model) #[batch_size*(spec_len/patch_size), 16, d_model]
        spec_16 = self.attention1(spec_16, spec_16, spec_16)
        spec_16 = spec_16.view(batch_size, -1, 16, self.d_model) #[batch_size, spec_len/16=200, 16, d_model]
        spec_16 = spec_16.view(batch_size, spec_16.shape[1], -1) #[batch_size, spec_len/16=200, 16*d_model]
        spec_16 = self.patch1(spec_16) #[batch_size, spec_len/16=200, d_model]

        spec_32 = spec_embed.view(batch_size, -1, 32, self.d_model) #[batch_size, spec_len/32=100, 32, d_model]
        spec_32 = spec_32.view(-1, 32, self.d_model) #[batch_size*(spec_len/patch_size), 32, d_model]
        spec_32 = self.attention2(spec_32, spec_32, spec_32)
        spec_32 = spec_32.view(batch_size, -1, 32, self.d_model) #[batch_size, spec_len/32=100, 32, d_model]
        spec_32 = spec_32.view(batch_size, spec_32.shape[1], -1) #[batch_size, spec_len/32=100, 32*d_model]
        spec_32 = self.patch2(spec_32) #[batch_size, spec_len/32=100, d_model]

        spec_64 = spec_embed.view(batch_size, -1, 64, self.d_model) #[batch_size, spec_len/64=50, 64, d_model]
        spec_64 = spec_64.view(-1, 64, self.d_model) #[batch_size*(spec_len/patch_size), 64, d_model]
        spec_64 = self.attention3(spec_64, spec_64, spec_64)
        spec_64 = spec_64.view(batch_size, -1, 64, self.d_model) #[batch_size, spec_len/64=50, 64, d_model]
        spec_64 = spec_64.view(batch_size, spec_64.shape[1], -1) #[batch_size, spec_len/64=50, 64*d_model]
        spec_64 = self.patch3(spec_64) #[batch_size, spec_len/64=50, d_model]

        spec_128 = spec_embed.view(batch_size, -1, 128, self.d_model) #[batch_size, spec_len/128=25, 128, d_model]
        spec_128 = spec_128.view(-1, 128, self.d_model) #[batch_size*(spec_len/patch_size), 128, d_model]
        spec_128 = self.attention4(spec_128, spec_128, spec_128)
        spec_128 = spec_128.view(batch_size, -1, 128, self.d_model) #[batch_size, spec_len/128=25, 128, d_model]
        spec_128 = spec_128.view(batch_size, spec_128.shape[1], -1) #[batch_size, spec_len/128=25, 128*d_model]
        spec_128 = self.patch4(spec_128) #[batch_size, spec_len/128=25, d_model]

        return torch.concat([spec_16, spec_32, spec_64, spec_128], dim=1) #[batch_size, 200+100+50+25=375, d_model]
    
class EmbedDiffPatch(nn.Module):
    def __init__(self, d_model=512, src_vocab=100) -> None:
        super(EmbedDiffPatch, self).__init__()
        """
        spectra length: 3200
        """
        # assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        # self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
        self.patch1 = nn.Linear(16*d_model, d_model)
        self.patch2 = nn.Linear(32*d_model, d_model)
        self.patch3 = nn.Linear(64*d_model, d_model)
        self.patch4 = nn.Linear(128*d_model, d_model)
        
    def forward(self, spec): #[batch_size, 1, spec_len]
        batch_size = spec.shape[0]
        spec_embed = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec_16 = spec_embed.view(batch_size, -1, 16, self.d_model) #[batch_size, spec_len/16=200, 16, d_model]
        spec_16 = spec_16.view(batch_size, spec_16.shape[1], -1) #[batch_size, spec_len/16=200, 16*d_model]
        spec_16 = self.patch1(spec_16) #[batch_size, spec_len/16=200, d_model]

        spec_32 = spec_embed.view(batch_size, -1, 32, self.d_model) #[batch_size, spec_len/32=100, 32, d_model]
        spec_32 = spec_32.view(batch_size, spec_32.shape[1], -1) #[batch_size, spec_len/32=100, 32*d_model]
        spec_32 = self.patch2(spec_32) #[batch_size, spec_len/32=100, d_model]
        
        spec_64 = spec_embed.view(batch_size, -1, 64, self.d_model) #[batch_size, spec_len/64=50, 64, d_model]
        spec_64 = spec_64.view(batch_size, spec_64.shape[1], -1) #[batch_size, spec_len/64=50, 64*d_model]
        spec_64 = self.patch3(spec_64) #[batch_size, spec_len/64=50, d_model]
        
        spec_128 = spec_embed.view(batch_size, -1, 128, self.d_model) #[batch_size, spec_len/128=25, 128, d_model]
        spec_128 = spec_128.view(batch_size, spec_128.shape[1], -1) #[batch_size, spec_len/128=25, 128*d_model]
        spec_128 = self.patch4(spec_128) #[batch_size, spec_len/128=25, d_model]
        
        return torch.concat([spec_16, spec_32, spec_64, spec_128], dim=1) #[batch_size, 200+100+50+25=375, d_model]

if __name__ == "__main__":
    # EmbedPatchDiffAttention
    # src_embed = EmbedPatchDiffAttention(d_model=512, src_vocab=100)
    src_embed = EmbedDiffPatch(d_model=512, src_vocab=100)
    
    data = torch.rand(5, 1, 3200)
    print(src_embed(data).shape)