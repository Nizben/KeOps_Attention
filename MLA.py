import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pykeops.torch import Genred

# -----------------------------
# 1) Standard multi-head latent attention
# -----------------------------
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, latent_dim, num_heads=8, dropout=0.0):
        """
        Standard multi-head latent attention (DeepSeek style) without mask.
        - dim: model dimension.
        - latent_dim: latent projection dimension.
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
        # x: (b, n, dim) -- note mask argument is ignored.
        b, n, dim = x.shape
        q = self.to_q(x)               # (b, n, dim)
        kv = self.to_kv(x)             # (b, n, 2*dim)
        k, v = kv.chunk(2, dim=-1)     # each (b, n, dim)
        k = self.latent_k(k)           # (b, n, dim)
        v = self.latent_v(v)           # (b, n, dim)
        # Multi-head reshape using view/transpose (without einops)
        q = q.view(b, n, self.num_heads, self.head_dim)  # (b, n, h, d)
        q = q.transpose(1, 2).contiguous() * self.scale   # (b, h, n, d)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, n, d)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, n, d)
        # Compute scaled dot-product scores (for standard latent attention)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (b, h, n, n)
        attn = F.softmax(scores, dim=-1)                # (b, h, n, n)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                     # (b, h, n, d)
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(b, n, self.num_heads * self.head_dim)
        return self.to_out(out)


# -----------------------------
# 2) KeOps-based multi-head latent attention
# -----------------------------
class KeOpsMultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, latent_dim, num_heads=8, dropout=0.0):
        """
        KeOps-enhanced multi-head latent attention without mask and using a single reduction operator.
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
        
        formula = '((q|k)/D)'
        aliases = [
            f'q = Vi({self.head_dim})',
            f'k = Vj({self.head_dim})',
            'D = Pm(1)',
            f'v = Vj({self.head_dim})'
        ]
        reduction_op = 'SumSoftMaxWeight'
        self.reduction = Genred(formula, aliases, reduction_op, formula2='v', axis=1)

    def forward(self, x, mask=None):
        # x: (b, n, dim)
        b, n, _ = x.shape
        q = self.to_q(x)              # (b, n, dim)
        kv = self.to_kv(x)            # (b, n, 2*dim)
        k, v = kv.chunk(2, dim=-1)    # each (b, n, dim)
        k = self.latent_k(k)          # (b, n, dim)
        v = self.latent_v(v)          # (b, n, dim)
        # Reshape to multi-head 
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous() * self.scale  # (b, h, n, d)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()              # (b, h, n, d)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()              # (b, h, n, d)
        # Merge batch and heads for KeOps.
        B = b * self.num_heads
        q = q.view(B, n, self.head_dim)  # (B, n, d)
        k = k.view(B, n, self.head_dim)  # (B, n, d)
        v = v.view(B, n, self.head_dim)  # (B, n, d)
        # Define scaling factor D
        D = torch.tensor([np.sqrt(self.head_dim)], dtype=x.dtype, device=x.device)
        # Use single Genred operator that computes SumSoftMaxWeight
        out_flat = self.reduction(q, k, D, v)   # (B, n, d)
        # Reshape back to (b, n, dim)
        out = out_flat.view(b, self.num_heads, n, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(b, n, self.num_heads * self.head_dim)
        out = self.dropout(out)
        return self.to_out(out)


# -----------------------------
# Benchmarking functions
# -----------------------------
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
    print(f"Seq Length: {n}, Vanilla MLA: {t_std:.2f}ms, KeOps MLA: {t_keops:.2f}ms")