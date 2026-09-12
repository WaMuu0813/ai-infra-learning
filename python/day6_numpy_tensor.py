import numpy as np


x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# print("x =")
# print(x)

# print("shape =", x.shape)
# print("ndim =", x.ndim)
# print("size =", x.size)

# print("\n--- reduction ---")

# print("sum all =", x.sum())
# print("sum axis=0 =", x.sum(axis=0))
# print("sum axis=1 =", x.sum(axis=1))

# print("\n--- indexing ---")

# print("x[0] =", x[0])
# print("x[1] =", x[1])

# print("x[0, 1] =", x[0, 1])

# print("x[:, 1] =", x[:, 1])
# print("x[0, :] =", x[0, :])

# print("\n--- reshape ---")

# y = np.arange(24)

# print("y =", y)
# print("y.shape =", y.shape)

# a = y.reshape(2, 3, 4)
# print("a.shape =", a.shape)

# b = a.reshape(6, 4)
# print("b.shape =", b.shape)

# c = a.reshape(2, -1)
# print("c.shape =", c.shape)

# print("\n--- broadcasting ---")

# a = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ])

# b = np.array([10, 20, 30])

# c = a + b

# print("a.shape =", a.shape)
# print("b.shape =", b.shape)
# print("c =")
# print(c)
# print("c.shape =", c.shape)

# print("\n--- element-wise vs matmul ---")

# A = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ])

# B = np.array([
#     [10, 20, 30],
#     [40, 50, 60]
# ])

# C = A * B

# print("A.shape =", A.shape)
# print("B.shape =", B.shape)
# print("A * B =")
# print(C)
# print("(A * B).shape =", C.shape)


# D = np.array([
#     [1, 2],
#     [3, 4],
#     [5, 6]
# ])

# E = A @ D

# print("\nD.shape =", D.shape)
# print("A @ D =")
# print(E)
# print("(A @ D).shape =", E.shape)

# print("\n--- mean and variance ---")

# x = np.array([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0]
# ])

# mean = x.mean(axis=1, keepdims=True)
# var = x.var(axis=1, keepdims=True)

# print("x.shape =", x.shape)

# print("mean =")
# print(mean)
# print("mean.shape =", mean.shape)

# print("var =")
# print(var)
# print("var.shape =", var.shape)

# normalized = (x - mean) / np.sqrt(var + 1e-5)

# print("normalized =")
# print(normalized)

# print("\n--- variance ---")

# v = np.array([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0]
# ])

# mean = v.mean(axis=1, keepdims=True)
# var = v.var(axis=1, keepdims=True)

# print("mean =")
# print(mean)

# print("variance =")
# print(var)

# print("mean.shape =", mean.shape)
# print("var.shape =", var.shape)

print("\n--- standardization ---")

x = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

mean = x.mean(axis=1, keepdims=True)
var = x.var(axis=1, keepdims=True)
std = np.sqrt(var)

normalized = (x - mean) / std

print("mean =")
print(mean)

print("var =")
print(var)

print("std =")
print(std)

print("normalized =")
print(normalized)

print("normalized mean =", normalized.mean(axis=1))
print("normalized var =", normalized.var(axis=1))