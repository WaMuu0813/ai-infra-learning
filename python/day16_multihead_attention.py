import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        B, T, D = x.shape

        # 1. 生成 Q / K / V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        print("after projection:")
        print("Q:", Q.shape)
        print("K:", K.shape)
        print("V:", V.shape)

        # 2. 把 d_model 拆成 num_heads × head_dim
        Q = Q.reshape(B, T, self.num_heads, self.head_dim)
        K = K.reshape(B, T, self.num_heads, self.head_dim)
        V = V.reshape(B, T, self.num_heads, self.head_dim)

        print("\nafter reshape:")
        print("Q:", Q.shape)

        # 3. 把 head 维放到前面
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        print("\nafter transpose:")
        print("Q:", Q.shape)

        # 4. 每个 head 独立计算 QK^T
        scores = Q @ K.transpose(-2, -1)

        # 5. scaled dot-product
        scores = scores / math.sqrt(self.head_dim)

        print("\nafter QK^T:")
        print("scores:", scores.shape)

        # 6. causal mask：屏蔽未来 token
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        print("\nmask shape:", mask.shape)

        scores = scores.masked_fill(mask, float("-inf"))

        # 7. 对每个 Query 的所有 Key 做 Softmax
        weights = torch.softmax(scores, dim=-1)

        print("weights shape:", weights.shape)

        # 8. 每个 head 根据 attention weights 加权 V
        out = weights @ V

        print("\nafter weights @ V:")
        print("out:", out.shape)

        # 9. 把 head 维度交换回去
        out = out.transpose(1, 2)

        print("\nafter transpose:")
        print("out:", out.shape)

        # 10. 合并所有 heads
        out = out.reshape(B, T, D)

        print("\nafter merge heads:")
        print("out:", out.shape)

        # 11. output projection
        out = self.out_proj(out)

        print("\nafter output projection:")
        print("out:", out.shape)

        return out


torch.manual_seed(0)

x = torch.randn(3, 10, 64)

attention = MultiHeadAttention(
    d_model=64,
    num_heads=8,
)

attention(x)