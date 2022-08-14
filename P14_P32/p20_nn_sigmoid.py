import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader

# torchvision 中的CIFAR10数据集
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

# 加载数据集
dataloader = DataLoader(dataset, batch_size=64)


# 搭建网络 Sigmoid
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.relu = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


# 创建网络
mymodule = MyModule()

# tensorboard
writer = SummaryWriter('p20_logs')

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, global_step=step)
    output = mymodule(imgs)
    writer.add_images('output', output, global_step=step)
    step = step + 1

writer.close()
