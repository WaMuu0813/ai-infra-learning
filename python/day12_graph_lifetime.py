import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y * 4

print("before backward:")
print("x.grad =", x.grad)
print("y.grad_fn =", y.grad_fn)
print("z.grad_fn =", z.grad_fn)

# z.backward()
z.backward(retain_graph=True)

print("\nafter first backward:")
print("x.grad =", x.grad)

print("\nsecond backward:")
z.backward()
print("x.grad =", x.grad)