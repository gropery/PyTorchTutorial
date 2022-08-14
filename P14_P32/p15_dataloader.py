import torchvision
from torch.utils.data import DataLoader

# 准备的测试数据集
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中的第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('p15_logs')
for epoch in range(2):
    step = 0
    for data in test_loader:     # 一个for循环即取完一次完整的数据
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('Epoch{}'.format(epoch), imgs, step)
        step = step + 1

writer.close()
