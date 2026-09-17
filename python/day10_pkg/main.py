# from mymodel.model import LinearModel
from mymodel import LinearModel


model = LinearModel(2)

print(model.forward(10))