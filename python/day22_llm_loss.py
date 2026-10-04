import torch
import torch.nn as nn

# B = batch size
# T = sequence length
# V = vocabulary size
B = 2
T = 4
V = 5

# 假装这是 Transformer forward 得到的 logits
logits = torch.randn(B, T, V)

# 假装这是两条 token 序列
labels = torch.tensor([
    [0, 1, 2, 3],
    [1, 2, -100, -100]
])

print("logits:", logits.shape)
print("labels:", labels.shape)

# next-token prediction 的位置对齐
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]

print("shift_logits:", shift_logits.shape)
print("shift_labels:", shift_labels.shape)

# 把 B 和 T 两个维度摊平成 N
flat_logits = shift_logits.reshape(-1, V)
flat_labels = shift_labels.reshape(-1)

print("flat_logits:", flat_logits.shape)
print("flat_labels:", flat_labels.shape)

criterion = nn.CrossEntropyLoss()
loss = criterion(flat_logits, flat_labels)

print("loss:", loss.item())