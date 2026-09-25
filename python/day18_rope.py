import math
import torch


def rotate_2d(x, theta):
    x1 = x[0]
    x2 = x[1]

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    y1 = x1 * cos_theta - x2 * sin_theta
    y2 = x1 * sin_theta + x2 * cos_theta

    return torch.stack([y1, y2])


x = torch.tensor([1.0, 0.0])

theta = math.pi / 2

y = rotate_2d(x, theta)

print("before:", x)
print("after: ", y)

base_theta = math.pi / 6  # 30°

for position in range(4):
    angle = position * base_theta
    y = rotate_2d(x, angle)

    print(
        f"position={position}, "
        f"angle={angle:.3f}, "
        f"vector={y}"
    )

head_dim = 8
base = 10000.0

indices = torch.arange(0, head_dim, 2)

print("indices:", indices)

inv_freq = 1.0 / (
    base ** (indices.float() / head_dim)
)

print("inv_freq:", inv_freq)

seq_len = 4

positions = torch.arange(seq_len)

print("positions:", positions)

angles = positions[:, None] * inv_freq[None, :]

print("angles shape:", angles.shape)
print("angles:")
print(angles)

cos = torch.cos(angles)
sin = torch.sin(angles)

print("cos shape:", cos.shape)
print("sin shape:", sin.shape)