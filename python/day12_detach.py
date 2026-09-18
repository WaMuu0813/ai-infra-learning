import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
y2 = y.detach()
z = y2 * 4

print("x =", x)
print("y =", y)
print("y.requires_grad =", y.requires_grad)
print("y.grad_fn =", y.grad_fn)

print()

print("y2 =", y2)
print("y2.requires_grad =", y2.requires_grad)
print("y2.grad_fn =", y2.grad_fn)

print()

print("z =", z)
print("z.requires_grad =", z.requires_grad)
print("z.grad_fn =", z.grad_fn)

# print("\ntry backward:")
# z.backward()

print("\n===== detach + requires_grad =====")

a = torch.tensor(2.0, requires_grad=True)
b = a * 3

c = b.detach()
c.requires_grad_(True)

d = c * 4

print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)

print("b.requires_grad =", b.requires_grad)
print("c.requires_grad =", c.requires_grad)
print("d.requires_grad =", d.requires_grad)

print("c.grad_fn =", c.grad_fn)
print("d.grad_fn =", d.grad_fn)

d.backward()

print("a.grad =", a.grad)
print("c.grad =", c.grad)