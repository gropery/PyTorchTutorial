import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

# 搭建网络 ReLU
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.relu = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.relu(input)
        return output

# 创建网络
mymodule = MyModule()
output = mymodule(input)
print(output)
