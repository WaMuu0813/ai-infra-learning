# print("mymodel.model is loading")
# from .utils import foo


class LinearModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x):
        return x * self.weight

print("__name__ =", __name__)
print("__package__ =", __package__)

from .utils import hello

hello()