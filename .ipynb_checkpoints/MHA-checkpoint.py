import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pykeops.torch import Genred

import time

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        """
        Standard multi-head self-attention.
        - dim: model (embedding) dimension.
        - num_heads: number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        # x: (b, n, dim)
        b, n, dim = x.shape
        qkv = self.to_qkv(x)  # (b, n, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads) * self.scale
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, n)
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class KeOpsMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        """
        Multi-head self-attention using KeOps Genred.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        # x: (b, n, dim)
        b, n, dim = x.shape
        qkv = self.to_qkv(x)  # (b, n, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to (b, heads, n, head_dim)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads) * self.scale
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Merge batch and head dims for KeOps:
        B = b * self.num_heads
        q = q.reshape(B, n, self.head_dim)  # (B, n, d)
        k = k.reshape(B, n, self.head_dim)  # (B, n, d)
        v = v.reshape(B, n, self.head_dim)  # (B, n, d)
        
        # Depending on whether a mask is provided, build the appropriate Genred operators.
        if mask is not None:
            # Expand mask: (b, n) -> (b, 1, n) and repeat for each head, then reshape to (B, n)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1).reshape(B, n)
            # Prepare mask tensor (as a feature with 1 dimension)
            m = mask.float().unsqueeze(-1)  # (B, n, 1)
            m = torch.where(m == 1.0,
                            torch.zeros_like(m),
                            torch.full_like(m, float('-inf')))
            # Set up Genred operators that incorporate the mask.
            attn_denom_genred = Genred(
                'Exp((q|k)+m)', 
                [
                    'q = Vi(0, {})'.format(self.head_dim),
                    'k = Vj(1, {})'.format(self.head_dim),
                    'm = Vj(2, 1)'
                ],
                'Sum',
                axis=1
            )
            attn_numer_genred = Genred(
                'Exp((q|k)+m)*v', 
                [
                    'q = Vi(0, {})'.format(self.head_dim),
                    'k = Vj(1, {})'.format(self.head_dim),
                    'm = Vj(2, 1)',
                    'v = Vj(3, {})'.format(self.head_dim)
                ],
                'Sum',
                axis=1
            )
            denom = attn_denom_genred(q, k, m)  # (B, n, 1)
            numer = attn_numer_genred(q, k, m, v)  # (B, n, head_dim)
        else:
            # No mask branch: simply compute exp(dot) weighted sum.
            attn_denom_genred = Genred(
                'Exp((q|k))', 
                [
                    'q = Vi(0, {})'.format(self.head_dim),
                    'k = Vj(1, {})'.format(self.head_dim)
                ],
                'Sum',
                axis=1
            )
            attn_numer_genred = Genred(
                'Exp((q|k))*v', 
                [
                    'q = Vi(0, {})'.format(self.head_dim),
                    'k = Vj(1, {})'.format(self.head_dim),
                    'v = Vj(2, {})'.format(self.head_dim)
                ],
                'Sum',
                axis=1
            )
            denom = attn_denom_genred(q, k)  # (B, n, 1)
            numer = attn_numer_genred(q, k, v)  # (B, n, head_dim)
        
        # Normalize the weighted sum to get the attention output.
        out = numer / (denom + 1e-6)
        out = out.reshape(b, self.num_heads, n, self.head_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.dropout(out)
        return self.to_out(out)



def benchmark(module, x, num_runs=50):
    # Warmup
    for _ in range(10):
        _ = module(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = module(x)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    return sum(times)/len(times)


seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
times_standard = []
times_keops = []

dim = 256
num_heads = 8
batch = 2
for n in seq_lengths:
    x = torch.randn(batch, n, dim, device='cuda')
    attn_std = MultiHeadAttention(dim, num_heads=num_heads).to('cuda')
    attn_keops = KeOpsMultiHeadAttention(dim, num_heads=num_heads).to('cuda')
    t_std = benchmark(attn_std, x)
    t_keops = benchmark(attn_keops, x)
    times_standard.append(t_std)
    times_keops.append(t_keops)