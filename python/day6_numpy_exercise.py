import numpy as np


x = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 4.0, 6.0, 8.0],
    [10.0, 20.0, 30.0, 40.0]
])

print("x =")
print(x)
print("x.shape =", x.shape)


mean = x.mean(axis=1, keepdims=True)
var = x.var(axis=1, keepdims=True)
std = np.sqrt(var + 1e-5)

y = (x - mean) / std

print("\nmean =")
print(mean)
print("mean.shape =", mean.shape)

print("\nvar =")
print(var)
print("var.shape =", var.shape)

print("\ny =")
print(y)

print("\ny mean =", y.mean(axis=1))
print("y var =", y.var(axis=1))