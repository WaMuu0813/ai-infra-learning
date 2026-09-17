import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        # 可训练参数
        self.weight = nn.Parameter(
            torch.tensor([2.0])
        )

        # 推理时需要的状态
        self.register_buffer(
            "bias",
            torch.tensor([1.0])
        )


    def forward(self, x):

        return x * self.weight + self.bias



def preprocess(x):

    # 模拟输入处理
    return x / 10



def postprocess(output):

    # 模拟结果处理
    return output.item()



model = SimpleModel()


# 模拟加载训练好的模型
checkpoint = model.state_dict()

print("model state:")
print(checkpoint)



# 输入
input_data = torch.tensor([50.0])


# 推理流程

model.eval()


with torch.inference_mode():

    x = preprocess(input_data)

    output = model(x)

    result = postprocess(output)


print("result:", result)