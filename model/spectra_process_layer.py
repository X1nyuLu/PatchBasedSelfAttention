import torch
import torch.nn as nn
# from torch.nn.functional import pad
# import math


# import sys
# sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/")
# import FinalResult.code.utils.the_annotated_transformer as atf
import utils.the_annotated_transformer as atf

class SpecDirectEmbed(nn.Module):
    def __init__(self, d_model=512, src_vocab=100) -> None:
        super(SpecDirectEmbed, self).__init__()
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
    def forward(self, spec):
        return self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]       

class EmbedPatchAttention(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(EmbedPatchAttention, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = atf.MultiHeadedAttention(h=8, d_model=512)
        
    def forward(self, spec): #[batch_size, spec_len]
        batch_size = spec.shape[0]
        spec = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(-1, self.patch_len, self.d_model) #[batch_size*(spec_len/patch_size), patch_size, d_model]
        spec = self.attention(spec, spec, spec)

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(batch_size, spec.shape[1], -1) #[batch_size, spec_len/patch_size, patch_size*d_model]
        
        return self.patch(spec) #[batch_size, spec_len/patch_size, d_model]

class EmbedPatchAttentionWithFullToken(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(EmbedPatchAttentionWithFullToken, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = atf.MultiHeadedAttention(h=8, d_model=d_model)

        self.full_linear = nn.Linear(int(spec_len/patch_len*d_model), d_model)
        
        self.full_attention = atf.MultiHeadedAttention(h=4, d_model=d_model)
        # self.full_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.full_token = nn.Parameter(torch.randn(1, 1, d_model)* math.sqrt(self.d_model))
    def forward(self, spec): #[batch_size, spec_len]
        batch_size = spec.shape[0]
        spec = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(-1, self.patch_len, self.d_model) #[batch_size*(spec_len/patch_size), patch_size, d_model]
        spec = self.attention(spec, spec, spec)

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(batch_size, spec.shape[1], -1) #[batch_size, spec_len/patch_size, patch_size*d_model]
        spec = self.patch(spec) #[batch_size, spec_len/patch_size, d_model]

        # full_token = self.full_token.expand(batch_size, -1, -1)
        # spec_ = spec.view(batch_size, 1, -1)

        spec_ = self.full_attention(spec, spec, spec)
        spec_ = spec_.view(batch_size, 1, -1)

        full_token = self.full_linear(spec_)
        
        # print(full_token.shape)

        spec_tokens = torch.cat((full_token, spec),dim=1) #[batch_size, 1+spec_len/patch_size, d_model]
        # print(spec_tokens.shape)
        
        return spec_tokens
    
class EmbedPatchAttentionWith3MoreTokens(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(EmbedPatchAttentionWith3MoreTokens, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = atf.MultiHeadedAttention(h=8, d_model=d_model)

        total_spec_token_num = int(spec_len/patch_len)
        self.full_linear = nn.Linear(int(total_spec_token_num*d_model), d_model)

        self.fingerprint_token_num = int((1350-400)/(3982-400)*total_spec_token_num)
        self.others_token_num = total_spec_token_num - self.fingerprint_token_num
        
        self.fingerprint_linear = nn.Linear(int(self.fingerprint_token_num*d_model), d_model)
        self.others_linear = nn.Linear(int(self.others_token_num*d_model), d_model)

        self.full_att = atf.MultiHeadedAttention(h=1, d_model=d_model)
        self.fingerprint_att = atf.MultiHeadedAttention(h=1, d_model=d_model)
        self.others_att = atf.MultiHeadedAttention(h=1, d_model=d_model)
        

        
    def forward(self, spec): #[batch_size, spec_len]
        batch_size = spec.shape[0]
        spec = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(-1, self.patch_len, self.d_model) #[batch_size*(spec_len/patch_size), patch_size, d_model]
        spec = self.attention(spec, spec, spec)

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(batch_size, spec.shape[1], -1) #[batch_size, spec_len/patch_size, patch_size*d_model]
        spec = self.patch(spec) #[batch_size, spec_len/patch_size, d_model]

        # full_token = self.full_token.expand(batch_size, -1, -1)
        
        spec = self.full_att(spec, spec, spec)
        spec_ = spec.view(batch_size, 1, -1)
        full_token = self.full_linear(spec_)

        finger_ = spec[:,:self.fingerprint_token_num,:]
        finger_ = self.fingerprint_att(finger_, finger_, finger_)
        finger_ = finger_.view(batch_size,1,-1)
        finger_token = self.fingerprint_linear(finger_)

        others_ = spec[:,self.fingerprint_token_num:,:]
        others_ = self.others_att(others_, others_, others_)
        others_ = others_.view(batch_size,1,-1)
        others_token = self.others_linear(others_)
        
        # print(full_token.shape)
        # print(finger_token.shape)
        # print(others_token.shape)

        spec_tokens = torch.cat((full_token, finger_token, others_token, spec),dim=1) #[batch_size, 3+spec_len/patch_size, d_model]
        # print(spec_tokens.shape)
        
        return spec_tokens

class EmbedPatchAttentionWith400Tokens(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(EmbedPatchAttentionWith400Tokens, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = atf.Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = atf.MultiHeadedAttention(h=8, d_model=d_model)
        
        # self.full_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, spec): #[batch_size, spec_len]
        batch_size = spec.shape[0]
        # indices = torch.arange(0, 3200, 8)
        indices = torch.arange(7, 3207, 8)
        # print(spec.shape)
        spec_400 = spec[:, indices]
        # print(spec_400.shape)
        spec_400 = self.embed(spec_400.to(torch.int)).squeeze(1) #[batch_size, 400, d_model]
        # print(spec_400.shape)
        # return 0
        return spec_400

        spec = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(-1, self.patch_len, self.d_model) #[batch_size*(spec_len/patch_size), patch_size, d_model]
        spec = self.attention(spec, spec, spec)

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(batch_size, spec.shape[1], -1) #[batch_size, spec_len/patch_size, patch_size*d_model]
        spec = self.patch(spec) #[batch_size, spec_len/patch_size, d_model]

        spec_tokens = torch.cat((spec_400, spec),dim=1)
        # print("spec_tokens: ",spec_tokens.shape)
               
        return spec_tokens

"""
ref
"""
    
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