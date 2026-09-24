import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return self.weight * (x / rms)

    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        out = weights @ V

        out = out.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)

        hidden = gate * up

        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_size):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model,num_heads)

        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model,intermediate_size)

    def forward(self, x):
        # Attention 子层
        norm_x = self.attn_norm(x)
        attn_out = self.attn(norm_x)
        x = x + attn_out

        # MLP 子层
        norm_x = self.mlp_norm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    T = 10
    D = 64

    x = torch.randn(B, T, D)

    block = TransformerBlock(
        d_model=D,
        num_heads=8,
        intermediate_size=256,
    )

    out = block(x)

    print("input shape: ", x.shape)
    print("output shape:", out.shape)