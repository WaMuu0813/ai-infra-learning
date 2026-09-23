import math
import torch

# 3 个 token，每个 Q/K/V 都是二维
Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [10.0, 10.0],
])

# ① 所有 Q 和所有 K 两两做点积
scores = Q @ K.transpose(-2, -1)

print("raw scores:")
print(scores)

# ② scaled dot-product
d_k = Q.shape[-1]
scores = scores / math.sqrt(d_k)

print("\nscaled scores:")
print(scores)

# ③ causal mask
T = Q.shape[0]

# triu(..., diagonal=1)：
# 取主对角线上方的位置，也就是“未来 token”
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1
)

print("\nmask:")
print(mask)

# masked_fill：
# mask 为 True 的位置，用 -inf 替换
scores = scores.masked_fill(mask, float("-inf"))

print("\nmasked scores:")
print(scores)

# ④ 每个 Query 在所有 Key 上做 Softmax
weights = torch.softmax(scores, dim=-1)

print("\nattention weights:")
print(weights)

# ⑤ 根据 attention weights 加权汇总 V
out = weights @ V

print("\nattention output:")
print(out)




import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # 同一个 x 经过三个不同的 Linear，产生 Q/K/V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # print("\nQ shape:", Q.shape)
        # print("K shape:", K.shape)
        # print("V shape:", V.shape)

        d_k = Q.shape[-1]

        # (B,T,D) @ (B,D,T) -> (B,T,T)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(d_k)

        # print("scores shape:", scores.shape)

        T = x.shape[1]

        # True 表示未来位置，需要屏蔽
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

        # 每个 Query 对所有 Key 做 softmax
        weights = torch.softmax(scores, dim=-1)

        # print("weights shape:", weights.shape)

        # (B,T,T) @ (B,T,D) -> (B,T,D)
        out = weights @ V

        # print("out shape:", out.shape)

        return out


torch.manual_seed(0)

x = torch.randn(2, 5, 16)

attention = SingleHeadAttention(d_model=16)
out = attention(x)

print("\nfinal output shape:", out.shape)

print("\n===== Training Experiment =====")

torch.manual_seed(42)

# 固定输入
x_train = torch.randn(2, 5, 16)

# 人为构造一个固定目标
target = torch.randn(2, 5, 16)

model = SingleHeadAttention(d_model=16)

# SGD：最基础的梯度下降优化器
# lr=0.1：learning rate（学习率），控制每次参数更新的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# MSE：让 model(x_train) 尽量接近 target
criterion = nn.MSELoss()

# 保存训练前 W_Q 的副本
q_before = model.q_proj.weight.detach().clone()

for step in range(101):
    # ① 前向传播
    out = model(x_train)

    # ② 计算最终 loss
    loss = criterion(out, target)

    # ③ 清掉上一轮残留的梯度
    optimizer.zero_grad()

    # ④ 反向传播
    loss.backward()

    # 第一次 backward 后，看看 Q/K/V 是否真的有梯度
    if step == 0:
        print("\nGradient check:")
        print("W_Q grad norm:", model.q_proj.weight.grad.norm().item())
        print("W_K grad norm:", model.k_proj.weight.grad.norm().item())
        print("W_V grad norm:", model.v_proj.weight.grad.norm().item())

    # ⑤ 根据梯度更新 W_Q/W_K/W_V
    optimizer.step()

    if step % 20 == 0:
        print(f"step={step:3d}, loss={loss.item():.6f}")

q_after = model.q_proj.weight.detach().clone()

print("\nParameter change:")
print(
    "W_Q changed:",
    not torch.allclose(q_before, q_after)
)

