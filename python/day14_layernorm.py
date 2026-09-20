import torch
import torch.nn as nn

x = torch.tensor([
    [2.0, 4.0, 6.0, 8.0],
    [1.0, 3.0, 5.0, 7.0],
])

ln = nn.LayerNorm(4)

y = ln(x)

# print("x:")
# print(x)

# print("\ny:")
# print(y)

# print("\nmean:")
# print(y.mean(dim=-1))

# print("\nvariance:")
# print(y.var(dim=-1, unbiased=False))

# print("\ngamma:")
# print(ln.weight)

# print("\nbeta:")
# print(ln.bias)


# def reference_layer_norm(x, gamma, beta, eps=1e-5):
#     mean = x.mean(dim=-1, keepdim=True)

#     var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

#     x_hat = (x - mean) / torch.sqrt(var + eps)

#     return gamma * x_hat + beta


# y_ref = reference_layer_norm(
#     x,
#     ln.weight,
#     ln.bias,
# )

# print("\nreference:")
# print(y_ref)

# print("\nPyTorch vs reference:")
# print(torch.allclose(y, y_ref, atol=1e-5))

# with torch.no_grad():
#     ln.weight.fill_(2.0)   # gamma = 2
#     ln.bias.fill_(3.0)     # beta = 3

# y2 = ln(x)

# print("\nafter changing gamma and beta:")
# print(y2)

# print("\nmean:")
# print(y2.mean(dim=-1))

# print("\nvariance:")
# print(y2.var(dim=-1, unbiased=False))

def reference_rms_norm(x, gamma, eps=1e-5):
    rms = torch.sqrt(
        (x ** 2).mean(dim=-1, keepdim=True) + eps
    )

    x_norm = x / rms

    return gamma * x_norm


gamma = torch.ones(4)

y_rms = reference_rms_norm(x, gamma)

print("\nRMSNorm reference:")
print(y_rms)

print("\nRMS of output:")
print(
    torch.sqrt(
        (y_rms ** 2).mean(dim=-1)
    )
)