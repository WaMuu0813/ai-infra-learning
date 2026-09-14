import torch

x = torch.arange(24).reshape(4, 6)

# print("x =")
# print(x)
# print("x.shape =", x.shape)
# print("x.stride =", x.stride())
# print("x.storage_offset =", x.storage_offset())
# print("x.is_contiguous =", x.is_contiguous())

# print("\n--- y = x[1:3, 1::2] ---")

# y = x[1:3, 1::2]

# print("y =")
# print(y)
# print("y.shape =", y.shape)
# print("y.stride =", y.stride())
# print("y.storage_offset =", y.storage_offset())
# print("y.is_contiguous =", y.is_contiguous())

print("\n--- transpose / reshape / contiguous ---")

a = x.T

# print("a =")
# print(a)
# print("a.shape =", a.shape)
# print("a.stride =", a.stride())
# print("a.storage_offset =", a.storage_offset())
# print("a.is_contiguous =", a.is_contiguous())

# print("\n--- data_ptr ---")

# print("x.data_ptr() =", x.data_ptr())
# print("a.data_ptr() =", a.data_ptr())

# b = a.contiguous()

# print("\n--- b = a.contiguous() ---")
# print("b.shape =", b.shape)
# print("b.stride =", b.stride())
# print("b.storage_offset =", b.storage_offset())
# print("b.is_contiguous =", b.is_contiguous())
# print("b.data_ptr() =", b.data_ptr())

print("\n--- reshape after transpose ---")

c = a.reshape(24)

print("c =", c)
print("c.shape =", c.shape)
print("c.stride =", c.stride())
print("c.is_contiguous =", c.is_contiguous())
print("c.data_ptr() =", c.data_ptr())