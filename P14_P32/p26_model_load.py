import torch
import torchvision
from torch import nn

# 方式1 -> 保存方式1-不仅保存了网络模型的结构，还保存了网络模型中的一些参数
# 打印模型的结构，也可通过调试查看参数也被加载进来了。model->classifier->Protected Attributes->_modules->'0'
model = torch.load('p26_vgg16_method1.pth')
# print(model)

# 方式2 - 加载模型
# vgg16 = torchvision.models.vgg16(weights=None)
# vgg16.load_state_dict(torch.load('p26_vgg16_method2.pth'))

# print(vgg16)
# model = torch.load('p26_vgg16_method2.pth')
# print(model)

# 陷阱
# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         output = self.conv1(x)
#         return output

from p26_model_save import *

model = torch.load('p26_mymodule_method1_pth')
print(model)
