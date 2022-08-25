import torch
import torchvision
from torch import nn

# weights=None 设置vgg 网络层中的参数是默认的，未经过训练的
vgg16 = torchvision.models.vgg16(weights=None)

##########################################################################
# 保存方式1-模型结构+模型参数
# (不仅保存了网络模型的结构，还保存了网络模型中的一些参数)
torch.save(vgg16, 'p26_vgg16_method1.pth')

##########################################################################
# 保存方式2-模型参数(官方推荐)
# (将网络模型vgg16中的参数保存成python字典形式)
torch.save(vgg16.state_dict(), 'p26_vgg16_method2.pth')

##########################################################################
# 方式1有陷阱-自定义的网络，需要在加载时包含
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        output = self.conv1(x)
        return output


mymodule = MyModule()
torch.save(mymodule, 'p26_mymodule_method1_pth')
##########################################################################
