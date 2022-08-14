import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


mymodule = MyModule()
x = torch.tensor(1.0)
output = mymodule(x)
print(output)
