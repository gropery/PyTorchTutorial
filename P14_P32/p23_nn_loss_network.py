import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

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
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
mymodule = MyModule()
for data in dataloader:
    imgs, targets = data
    outputs = mymodule(imgs)

    # 可以看下output和targets长什么样，看选择什么样的损失函数
    # print(outputs)
    # print(targets)

    result_loss = loss(outputs, targets)
    # print(result_loss)

    # 反向传播 梯度下降
    # 查看 mymodule->modul1->Protected Attributes->_modules->'0'->weight->grad
    result_loss.backward()
    print('ok')
