import torch
from torch import nn


layer = nn.Linear(3, 2)

print("layer:")
print(layer)

print("\nweight:")
print(layer.weight)
print("weight shape:", layer.weight.shape)

print("\nbias:")
print(layer.bias)
print("bias shape:", layer.bias.shape)

print("\nparameters:")
for name, p in layer.named_parameters():
    print(name, p.shape)


x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
])

y = layer(x)

print("\nx shape:", x.shape)
print("y shape:", y.shape)
print("y:")
print(y)

print("\ny grad_fn:", y.grad_fn)