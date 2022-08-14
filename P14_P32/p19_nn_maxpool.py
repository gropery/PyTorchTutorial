import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

# 自定义一个网络，网络中有最大池化的功能
from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.maxpoll1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpoll1(input)
        return output


# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input,(-1, 1, 5, 5))
# print(input.shape)
#
# mymodule = MyModule()
# output = mymodule(input)
# print(output)

dataset = torchvision.datasets.CIFAR10('dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

mymodule = MyModule()

writer = SummaryWriter('p19_logs')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output = mymodule(imgs)
    writer.add_images('output', output, step)
    step = step + 1

writer.close()
