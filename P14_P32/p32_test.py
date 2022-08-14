import torch
import torchvision.transforms
from PIL import Image
from torch import nn

# image_path = '../P6_10/imgs/dog.png'
image_path = '../P6_10/imgs/airplane.png'
image = Image.open(image_path)
print(image)

# png 格式图片除了RGB外，还有透明度通道，这里只保留颜色通道
image = image.convert('RGB')

# 改变图片尺寸->转换图片PIL格式为tensor类型
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


# 搭建神经网络
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x


# 加载训练好的网络
# model = torch.load('./p30_pth/p30_mymodule_29.pth')
model = torch.load('./p30_pth/p30_mymodule_29.pth', map_location=torch.device('cpu'))  # GPU 训练的网络映射到cpu中测试
print(model)

# 输入图片使用网络进行测试
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

# airplane = 0
# automobile = 1
# bird = 2
# cat = 3
# deer = 4
# dog = 5
# frog = 6
# horse = 7
# ship = 8
# truck = 9
print(output.argmax(1))
