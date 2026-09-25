import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, intermediate_size)
        self.up_proj = nn.Linear(d_model, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, d_model)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        activated_gate = gate * torch.sigmoid(gate)

        return self.down_proj(activated_gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_size):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, num_heads)

        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, intermediate_size)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))

        x = x + self.mlp(self.mlp_norm(x))

        return x


def apply_rope(q, k):
    B, H, T, Dh = q.shape

    indices = torch.arange(0, Dh, 2, dtype=q.dtype, device=q.device)

    inv_freq = 1.0 / (10000.0 ** (indices / Dh))

    positions = torch.arange(T, dtype=q.dtype, device=q.device)

    angles = positions[:, None] * inv_freq[None, :]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]

    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]

    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos

    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_out[..., 0::2] = q_rot_even
    q_out[..., 1::2] = q_rot_odd

    k_out[..., 0::2] = k_rot_even
    k_out[..., 1::2] = k_rot_odd

    return q_out, k_out


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim % 2 == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Day18 新增
        q, k = apply_rope(q, k)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim**0.5)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        out = weights @ v

        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, D)

        return self.out_proj(out)


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        intermediate_size,
        num_layers,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        # 1. token_ids 经过 embedding
        x = self.embedding(token_ids)

        # 2. 依次经过 self.layers 中每个 block
        for block in self.layers:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    model = TinyTransformer(
        vocab_size=1000,
        d_model=64,
        num_heads=8,
        intermediate_size=256,
        num_layers=2,
    )

    token_ids = torch.tensor([
        [10, 25, 30, 42, 18],
        [7, 11, 90, 3, 55],
    ])

    logits = model(token_ids)

    print("token_ids shape:", token_ids.shape)
    print("logits shape:", logits.shape)
    
targets = torch.tensor([
    [25, 30, 42, 18, 99],
    [11, 90, 3, 55, 21],
])

loss_fn = nn.CrossEntropyLoss()

B, T, V = logits.shape

loss = loss_fn(
    logits.reshape(B * T, V),
    targets.reshape(B * T)
)

print("loss:", loss.item())