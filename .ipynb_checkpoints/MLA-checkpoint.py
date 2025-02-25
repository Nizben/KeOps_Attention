import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time

from pykeops.torch import Genred

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, latent_dim, num_heads=8, dropout=0.0):
        """
        Standard multi-head latent attention (DeepSeek style).
        - dim: model dimension.
        - latent_dim: latent dimension for keys/values.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.latent_k = nn.Sequential(
            nn.Linear(dim, latent_dim, bias=False),
            nn.ReLU(),
            nn.Linear(latent_dim, dim, bias=False)
        )
        self.latent_v = nn.Sequential(
            nn.Linear(dim, latent_dim, bias=False),
            nn.ReLU(),
            nn.Linear(latent_dim, dim, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        b, n, dim = x.shape
        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=-1)
        k = self.latent_k(k)
        v = self.latent_v(v)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads) * self.scale
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class KeOpsMultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, latent_dim, num_heads=8, dropout=0.0):
        """
        KeOps-enhanced multi-head latent attention.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.latent_k = nn.Sequential(
            nn.Linear(dim, latent_dim, bias=False),
            nn.ReLU(),
            nn.Linear(latent_dim, dim, bias=False)
        )
        self.latent_v = nn.Sequential(
            nn.Linear(dim, latent_dim, bias=False),
            nn.ReLU(),
            nn.Linear(latent_dim, dim, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        # x: (b, n, dim)
        b, n, _ = x.shape
        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=-1)
        
        # Apply latent projection to k and v.
        k = self.latent_k(k)
        v = self.latent_v(v)
        
        # Reshape to (b, num_heads, n, head_dim) and scale queries.
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads) * self.scale
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Merge batch and heads for KeOps.
        B = b * self.num_heads
        q = q.reshape(B, n, self.head_dim)  # (B, n, d)
        k = k.reshape(B, n, self.head_dim)  # (B, n, d)
        v = v.reshape(B, n, self.head_dim)  # (B, n, d)
        
        # Build Genred operators.
        if mask is not None:
            # Expand mask: (b, n) -> (B, n)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1).reshape(B, n)
            m = mask.float().unsqueeze(-1)  # (B, n, 1)
            m = torch.where(m == 1.0,
                            torch.zeros_like(m),
                            torch.full_like(m, float('-inf')))
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

# Benchmark latent attention modules.
seq_lengths_latent = [128, 256, 512, 1024, 2048, 4096, 8192]
times_latent_standard = []
times_latent_keops = []

dim = 256
latent_dim = 128  # chosen latent dimension
num_heads = 8
batch = 2

for n in seq_lengths_latent:
    x = torch.randn(batch, n, dim, device='cuda')
    attn_std = MultiHeadLatentAttention(dim, latent_dim, num_heads=num_heads).to('cuda')
    attn_keops = KeOpsMultiHeadLatentAttention(dim, latent_dim, num_heads=num_heads).to('cuda')
    t_std = benchmark(attn_std, x)
    t_keops = benchmark(attn_keops, x)
    times_latent_standard.append(t_std)
    times_latent_keops.append(t_keops)




# import matplotlib.pyplot as plt
# from MHA import seq_lengths, times_standard, times_keops
# from MLA import seq_lengths_latent, times_latent_standard, times_latent_keops
# plt.figure(figsize=(8,5))
# plt.plot(seq_lengths, times_standard, marker='o', label='Standard Attention')
# plt.plot(seq_lengths, times_keops, marker='s', label='KeOps Attention')
# plt.xlabel("Sequence Length")
# plt.ylabel("Forward Pass Time (ms)")
# plt.title("Runtime vs Sequence Length")
# plt.legend()
# plt.show()
# plt.figure(figsize=(8,5))
# plt.plot(seq_lengths_latent, times_latent_standard, marker='o', label='Standard Latent Attention')
# plt.plot(seq_lengths_latent, times_latent_keops, marker='s', label='KeOps Latent Attention')
# plt.xlabel("Sequence Length")
# plt.ylabel("Forward Pass Time (ms)")
# plt.title("Runtime vs Sequence Length (Latent Attention)")
# plt.legend()
# plt.show()