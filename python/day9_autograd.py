import torch


w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

a = w * x
b = a + 2
loss = b ** 2

print("=== before backward ===")
print("w =", w)
print("a =", a)
print("b =", b)
print("loss =", loss)

print("w.requires_grad =", w.requires_grad)
print("x.requires_grad =", x.requires_grad)
print("a.requires_grad =", a.requires_grad)

print("w.is_leaf =", w.is_leaf)
print("a.is_leaf =", a.is_leaf)

print("a.grad_fn =", a.grad_fn)
print("b.grad_fn =", b.grad_fn)
print("loss.grad_fn =", loss.grad_fn)

print("w.grad =", w.grad)

loss.backward()

print("\n=== after backward ===")
print("w.grad =", w.grad)

print("\n=== second backward ===")

a2 = w * x
b2 = a2 + 2
loss2 = b2 ** 2

loss2.backward()

print("w.grad =", w.grad)

print("\n=== branch graph ===")

w.grad.zero_()

a3 = w ** 2
b3 = 3 * w
loss3 = a3 + b3

print("a3 =", a3)
print("b3 =", b3)
print("loss3 =", loss3)

loss3.backward()

print("w.grad =", w.grad)