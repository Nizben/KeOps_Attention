import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pykeops.torch import Genred


class VanillaMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Project input to Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape and transpose for multi-head: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)

        # Compute weighted sum: (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(weights, V)

        # Reshape back: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(output)
        return output


class KeOpsMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections (same as vanilla)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Define KeOps reduction with SumSoftMaxWeight
        formula = '((Q|K)/D)'
        aliases = [
            f'Q = Vi({self.d_k})',  # Query vectors, i-dimension (seq_len)
            f'K = Vj({self.d_k})',  # Key vectors, j-dimension (seq_len)
            f'V = Vj({self.d_k})',  # Value vectors, j-dimension (seq_len)
            'D = Pm(1)',           # Scaling factor (scalar)
        ]
        formula2 = 'V'
        reduction_op = 'SumSoftMaxWeight'
        self.reduction = Genred(formula, aliases, reduction_op=reduction_op, formula2=formula2, axis=1)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Project input to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Flatten batch and heads: (batch_size * num_heads, seq_len, d_k)
        Q_flat = Q.reshape(batch_size * self.num_heads, seq_len, self.d_k)
        K_flat = K.reshape(batch_size * self.num_heads, seq_len, self.d_k)
        V_flat = V.reshape(batch_size * self.num_heads, seq_len, self.d_k)

        # Define scaling factor D = sqrt(d_k)
        D = torch.tensor([np.sqrt(self.d_k)], dtype=x.dtype, device=x.device)

        # Apply KeOps reduction: (batch_size * num_heads, seq_len, d_k)
        output_flat = self.reduction(Q_flat, K_flat, V_flat, D)

        # Reshape back: (batch_size, seq_len, d_model)
        output = output_flat.view(batch_size, self.num_heads, seq_len, self.d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(output)
        return output


# Set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Parameters for correctness check
d_model = 64
num_heads = 8
batch_size = 2
seq_len = 10

# Initialize models with the same weights
torch.manual_seed(0)
vanilla_attn = VanillaMultiHeadAttention(d_model, num_heads).to(device)
keops_attn = KeOpsMultiHeadAttention(d_model, num_heads).to(device)
keops_attn.load_state_dict(vanilla_attn.state_dict())  # Copy weights

# Create input
x = torch.randn(batch_size, seq_len, d_model).to(device)

# Compute outputs
with torch.no_grad():
    output_vanilla = vanilla_attn(x)
    output_keops = keops_attn(x)

# Verify correctness
is_correct = torch.allclose(output_vanilla, output_keops, atol=1e-6)
print(f"Outputs match: {is_correct}")

# Benchmark with larger input
seq_len_large = 1024
batch_size_large = 16
x_large = torch.randn(batch_size_large, seq_len_large, d_model).to(device)

# Warm-up runs to account for compilation/initialization
for _ in range(10):
    vanilla_attn(x_large)
    keops_attn(x_large)
torch.cuda.synchronize()

# Benchmark vanilla
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for _ in range(10):
    vanilla_attn(x_large)
end_event.record()
torch.cuda.synchronize()
time_vanilla = start_event.elapsed_time(end_event) / 10  # Average time in ms

# Benchmark KeOps
start_event.record()
for _ in range(10):
    keops_attn(x_large)
end_event.record()
torch.cuda.synchronize()
time_keops = start_event.elapsed_time(end_event) / 10  # Average time in ms

print(f"Vanilla average time: {time_vanilla:.4f} ms")
print(f"KeOps average time: {time_keops:.4f} ms")
