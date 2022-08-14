import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

# 这个数据集太大了140多G
# train_data = torchvision.datasets.ImageNet('data_image_net', split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

train_data = torchvision.datasets.CIFAR10('dataset', train=True, transform=torchvision.transforms.ToTensor,
                                          download=True)

# vgg 中weights为None时，只是加载网络模型，参数使用默认的
#     中weights为DEFAULT时，会从网络中下载每个层的参数(这些参数是在ImageNet数据集中已经训练好的)
# 下载文件会放在C:\Users\gropery\.cache\torch\hub\checkpoints\vgg16-397923af.pth     527MB
vgg16_false = torchvision.models.vgg16(weights=None)
#  The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`.
#  You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

print('1.vgg16_true-----------------------------------')
print(vgg16_true)
# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print('2.vgg16_true-----------------------------------')
print(vgg16_true)

print('3.vgg16_false-----------------------------------')
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print('4.vgg16_false-----------------------------------')
print(vgg16_false)
