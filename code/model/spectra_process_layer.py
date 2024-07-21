import torch
import torch.nn as nn

from ..utils import Embeddings, MultiHeadedAttention

class SpecDirectEmbed(nn.Module):
    def __init__(self, d_model=512, src_vocab=100) -> None:
        super(SpecDirectEmbed, self).__init__()
        self.d_model = d_model
        self.embed = Embeddings(d_model, src_vocab)
    def forward(self, spec):
        return self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]       

class EmbedPatchAttention(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=100) -> None:
        super(EmbedPatchAttention, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = MultiHeadedAttention(h=8, d_model=512)
        
    def forward(self, spec): #[batch_size, spec_len]
        batch_size = spec.shape[0]
        spec = self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(-1, self.patch_len, self.d_model) #[batch_size*(spec_len/patch_size), patch_size, d_model]
        spec = self.attention(spec, spec, spec)

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(batch_size, spec.shape[1], -1) #[batch_size, spec_len/patch_size, patch_size*d_model]
        
        return self.patch(spec) #[batch_size, spec_len/patch_size, d_model]
