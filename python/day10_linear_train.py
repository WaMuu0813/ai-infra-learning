import torch
from torch import nn


model = nn.Linear(1, 1)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

y_true = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
])

for step in range(20):
    optimizer.zero_grad()

    prediction = model(x)

    loss = criterion(prediction, y_true)

    loss.backward()

    optimizer.step()

    print(
        f"step={step:02d}, "
        f"loss={loss.item():.6f}, "
        f"w={model.weight.item():.4f}, "
        f"b={model.bias.item():.4f}"
    )