import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


# Conv2d 公式计算经验总结
# (1) 如果输入和输出的尺寸没变，分母stride步幅参数应该为1，否则padding填充参数需要很大才能使得分子符合
# (2) 奇数卷积核把中心格子对准图片的第一个格子，卷积核在格子外有2层，padding=2
# 综上，如果尺寸不变，stride=1, padding=2
from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.model1(x)
        return x


mymodule = MyModule()
print(mymodule)

# bach_size 为64
input = torch.ones((64, 3, 32, 32))
output = mymodule(input)
# 1个图片输出10个结果，64个图片，输出[64,10]
print(output.shape)

writer = SummaryWriter('p22_logs')
writer.add_graph(mymodule, input)
writer.close()
