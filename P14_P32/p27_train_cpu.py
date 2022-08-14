import torchvision
from torch.utils.tensorboard import SummaryWriter
from p27_model import *
from torch import nn
from torch.utils.data import DataLoader
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集长度为: {}'.format(train_data_size))
print('测试数据集长度为: {}'.format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
mymodule = MyModule()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# leaning_rate = 0.01
# 1e-2 = 1x(10)^(-2) = 1/100 = 0.01
leaning_rate = 1e-2
optimizer = torch.optim.SGD(params=mymodule.parameters(), lr=leaning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_tran_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('p27_logs')

start_time = time.time()
for i in range(epoch):
    print('-----------第 {} 轮训练开始----------'.format(i+1))

    # 训练步骤开始
    mymodule.train()  # 对某些网络层有效 eg: Dropout,BatchNorm
    for data in train_dataloader:
        start_time = time.time()
        imgs, targets = data
        outputs = mymodule(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_tran_step = total_tran_step + 1
        if total_tran_step % 100 == 0:
            end_time = time.time()
            print('训练时间: {}'.format(end_time-start_time))
            print('训练次数: {}, Loss: {}'.format(total_tran_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_tran_step)

    # 测试步骤开始
    mymodule.eval()  # 对某些网络有效 eg: Dropout,BatchNorm
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 测试时，不需要网络模型中的梯度，不需要对梯度调整，更不需要梯度来优化
        for data in test_dataloader:
            imgs, targets = data
            outputs = mymodule(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print('整体测试集上的loss: {}'.format(total_test_loss))
    print('整体测试集上的正确率: {}'.format(total_accuracy/test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(mymodule, './p27_pth/p27_mymodule_{}.pth'.format(i)) # 方式1
    # torch.save(mymodule.state_dict(), './p27_pth/p27_mymodule_{}.pth'.format(i)) # 方式2-官方推荐
    print('模型已保存')

writer.close()

