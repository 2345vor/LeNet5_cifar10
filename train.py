import time
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from net import cifar_model
import os
# 定义训练的设备
# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=False)

# length 长度
train_data_size = len(train_data)  # 50000
test_data_size = len(test_data)  # 10000
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用Dataloader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# 加载模型并使用GPU
cifarr = cifar_model()
cifarr = cifarr.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(cifarr.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

# 添加tensorboard
folder = 'cifar_log'
if not os.path.exists(folder):
    os.mkdir('cifar_log')
writer = SummaryWriter("./cifar_log")
start_time = time.time()
min_acc = 0
for i in range(epoch):
    print("————————第{}轮训练开始————————".format(i+1))

    # 训练步骤开始
    cifarr.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = cifarr(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 优化器梯度清零
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度和优化器，更新权重

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 验证步骤
    cifarr.eval()
    total_test_loss = 0
    total_accuracy = 0  # 总体的准确率
    with torch.no_grad():  # 强制之后的内容不进行计算图构建
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = cifarr(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    a = total_accuracy/test_data_size
    # 保存最好的模型权重文件
    if a > min_acc:
        folder = 'cifar_weight'
        if not os.path.exists(folder):
            os.mkdir('cifar_weight')
        min_acc = a
        print('save best model', )
        torch.save(cifarr.state_dict(), "cifar_weight/best_model.pth")
    # 保存最后的权重文件
    if i == epoch - 1:
        torch.save(cifarr.state_dict(), "cifar_weight/last_model.pth")

writer.close()
