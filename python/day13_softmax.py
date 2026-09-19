import torch

# x = torch.tensor([1.0, 2.0, 3.0])

# y = torch.softmax(x, dim=0)

# print("x =", x)
# print("softmax =", y)
# print("sum =", y.sum())

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

def naive_softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def stable_softmax(x):
    max_x = x.max(dim=-1, keepdim=True).values
    shifted_x = x - max_x

    exp_x = torch.exp(shifted_x)
    sum_exp = exp_x.sum(dim=-1, keepdim=True)

    return exp_x / sum_exp

x_big = torch.tensor([
    [1000.0, 1001.0, 1002.0],
    [2000.0, 2001.0, 2002.0]
])

# print("\nnaive softmax:")
# print(naive_softmax(x_big))

# print("\ntorch softmax:")
# print(torch.softmax(x_big, dim=-1))

print("\nstable softmax:")
print(stable_softmax(x_big))

print("\ntorch softmax:")
print(torch.softmax(x_big, dim=-1))

print("\nclose?")
print(torch.allclose(
    stable_softmax(x_big),
    torch.softmax(x_big, dim=-1)
))



# y0 = torch.softmax(x, dim=0)
# y1 = torch.softmax(x, dim=1)

# print("x:")
# print(x)

# print("\ndim=0:")
# print(y0)

# print("\ndim=1:")
# print(y1)

# print("\ny0 sum over dim=0:")
# print(y0.sum(dim=0))

# print("\ny1 sum over dim=1:")
# print(y1.sum(dim=1))