import torch
import torch.nn as nn


# class DemoModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # 注册参数：参与 model.parameters()
#         self.weight = nn.Parameter(
#             torch.tensor([2.0])
#         )

#         # 注册 buffer：属于模型状态，但不是可训练参数
#         self.register_buffer(
#             "scale",
#             torch.tensor([3.0])
#         )

#         # 普通 Tensor 属性
#         self.tmp = torch.tensor([100.0])

#     def forward(self, x):
#         return x * self.weight * self.scale


# model = DemoModel()

# print("=== named_parameters ===")
# for name, value in model.named_parameters():
#     print(name, value)

# print("\n=== named_buffers ===")
# for name, value in model.named_buffers():
#     print(name, value)

# print("\n=== state_dict ===")
# for name, value in model.state_dict().items():
#     print(name, value)


class ModeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(
            p=0.5
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

model = ModeModel()

x = torch.randn(1,4)

print("=== train mode ===")

model.train()

print(model.training)

y1 = model(x)
y2 = model(x)

print(torch.equal(y1, y2))


print("=== eval mode ===")

model.eval()

print(model.training)

y3 = model(x)
y4 = model(x)

print(torch.equal(y3, y4))


print("\n=== no_grad test ===")

model.eval()

with torch.no_grad():

    y5 = model(x)

print(y5.requires_grad)


print("\n=== inference_mode test ===")

model.eval()

with torch.inference_mode():

    y6 = model(x)


print(y6.requires_grad)