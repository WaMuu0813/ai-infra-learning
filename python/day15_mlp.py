import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([
    [-2.0, 1.0, 3.0, -1.0],
    [1.0, 2.0, -1.0, 4.0],
])

# linear1 = nn.Linear(4, 8)
# relu = nn.ReLU()
# linear2 = nn.Linear(8, 4)

# h = linear1(x)
# a = relu(h)
# y = linear2(a)

# print("x.shape =", x.shape)
# print("h.shape =", h.shape)
# print("a.shape =", a.shape)
# print("y.shape =", y.shape)

# print("\nh =", h)
# print("\na =", a)


# print("\n--- token MLP ---")

# x3d = torch.randn(8, 128, 768)

# mlp = nn.Sequential(
#     nn.Linear(768, 3072),
#     nn.ReLU(),
#     nn.Linear(3072, 768),
# )

# y3d = mlp(x3d)

# print("input shape :", x3d.shape)
# print("output shape:", y3d.shape)


# values = torch.tensor([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])

# relu_out = torch.relu(values)
# silu_out = torch.nn.functional.silu(values)

# print("\n--- ReLU vs SiLU ---")
# print("x    =", values)
# print("ReLU =", relu_out)
# print("SiLU =", silu_out)


# print("\n--- SwiGLU MLP ---")

# x_llm = torch.randn(2, 4, 768)

# gate_proj = nn.Linear(768, 2048, bias=False)
# up_proj = nn.Linear(768, 2048, bias=False)
# down_proj = nn.Linear(2048, 768, bias=False)

# gate = torch.nn.functional.silu(gate_proj(x_llm))
# up = up_proj(x_llm)

# hidden = gate * up
# output = down_proj(hidden)

# print("x      :", x_llm.shape)
# print("gate   :", gate.shape)
# print("up     :", up.shape)
# print("hidden :", hidden.shape)
# print("output :", output.shape)

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False
        )

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        hidden = gate * up
        output = self.down_proj(hidden)

        return output


model = SwiGLUMLP(
    hidden_size=768,
    intermediate_size=2048
)

# x_module = torch.randn(2, 4, 768)
# y_module = model(x_module)

# print("\n--- SwiGLUMLP ---")
# print("input :", x_module.shape)
# print("output:", y_module.shape)


print("\n--- parameter count ---")

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("total parameters:", total_params)

print("\n--- parameter details ---")

for name, param in model.named_parameters():
    print(
        name,
        "shape =", tuple(param.shape),
        "numel =", param.numel()
    )

total_bytes = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)

print("\n--- parameter memory ---")
print("bytes:", total_bytes)
print("MiB:", total_bytes / 1024**2)


print("\n--- convert to bfloat16 ---")

model = model.to(torch.bfloat16)

bf16_total_params = sum(
    p.numel()
    for p in model.parameters()
)

bf16_total_bytes = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)

print("dtype:", next(model.parameters()).dtype)
print("total parameters:", bf16_total_params)
print(
    "element size:",
    next(model.parameters()).element_size(),
    "bytes"
)
print("parameter memory:", bf16_total_bytes / 1024**2, "MiB")