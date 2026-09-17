import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.tensor(0.0))

        # self.a = torch.tensor(10.0,requires_grad=True)

    def forward(self, x):
        return self.w * x


model = SimpleModel()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

x = torch.tensor(2.0)
target = torch.tensor(10.0)

for step in range(10):
    optimizer.zero_grad()

    prediction = model(x)
    loss = (prediction - target) ** 2

    loss.backward()

    print(
        f"step={step}, "
        f"w={model.w.item():.4f}, "
        f"prediction={prediction.item():.4f}, "
        f"loss={loss.item():.4f}, "
        f"grad={model.w.grad.item():.4f}"
    )

    optimizer.step()

# print("model =", model)
# print("w =", model.w)
# print("w.requires_grad =", model.w.requires_grad)

# print("\nparameters:")
# for p in model.parameters():
#     print(p)


# print("\na =", model.a)
# print("a.requires_grad =", model.a.requires_grad)

# print("\nnamed parameters:")
# for name, p in model.named_parameters():
#     print(name, p)