import torch
import torchvision
from torch import nn

##########################################################################
# 方式1 -> 保存方式1-不仅保存了网络模型的结构，还保存了网络模型中的一些参数
# 打印模型的结构，也可通过调试查看参数也被加载进来了。model->classifier->Protected Attributes->_modules->'0'
model = torch.load('p26_vgg16_method1.pth')
print(model)

##########################################################################
# 方式2 - 加载模型
vgg16 = torchvision.models.vgg16(weights=None)                # 新建网络模型的结构
vgg16.load_state_dict(torch.load('p26_vgg16_method2.pth'))    # 将之前保存的状态参数导入新模型中
print(vgg16)
#
# 以下打印可看出，方式二的model保存的是字典形式，内容为模型的状态参数
# model = torch.load('p26_vgg16_method2.pth')
# print(model)

##########################################################################
# 方式1有陷阱-自定义的网络，需要在加载时包含
# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         output = self.conv1(x)
#         return output
#
# 或者import定义的文件，也即包含
# from p26_model_save import *
#
# model = torch.load('p26_mymodule_method1.pth')
# print(model)
##########################################################################
