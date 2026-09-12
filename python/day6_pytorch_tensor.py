import torch


x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

# print("x =")
# print(x)

# print("shape =", x.shape)
# print("ndim =", x.ndim)
# print("numel =", x.numel())
# print("dtype =", x.dtype)

# print("\n--- dtype ---")

# a = torch.tensor([1, 2, 3])
# b = torch.tensor([1.0, 2.0, 3.0])
# c = torch.tensor([1, 2, 3], dtype=torch.float32)

# print("a =", a)
# print("a.dtype =", a.dtype)

# print("b =", b)
# print("b.dtype =", b.dtype)

# print("c =", c)
# print("c.dtype =", c.dtype)

# print("\n--- reshape ---")

# x = torch.tensor([
#     [1., 2., 3.],
#     [4., 5., 6.]
# ])

# y = x.reshape(3, 2)
# z = x.reshape(-1)

# print("x.shape =", x.shape)
# print("y.shape =", y.shape)
# print("z.shape =", z.shape)

# print("y =")
# print(y)

# print("z =")
# print(z)

# print("\n--- reduction and broadcasting ---")

# x = torch.tensor([
#     [1., 2., 3.],
#     [4., 5., 6.]
# ])

# mean = x.mean(dim=1, keepdim=True)
# centered = x - mean

# print("x =")
# print(x)

# print("mean =")
# print(mean)
# print("mean.shape =", mean.shape)

# print("centered =")
# print(centered)
# print("centered.shape =", centered.shape)

print("\n--- device ---")

x = torch.tensor([1., 2., 3.])

print("x =", x)
print("x.device =", x.device)
print("CUDA available =", torch.cuda.is_available())