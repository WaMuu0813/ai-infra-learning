import torch

w = torch.tensor(0.0, requires_grad=True)

target = 5.0

optimizer = torch.optim.SGD([w], lr=0.1)

for step in range(10):
    optimizer.zero_grad()

    loss = (w - target) ** 2

    loss.backward()

    print(
        f"step={step}, "
        f"w={w.item():.4f}, "
        f"loss={loss.item():.4f}, "
        f"grad={w.grad.item():.4f}"
    )

    optimizer.step()