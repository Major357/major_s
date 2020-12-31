import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from major_utils import set_seed
import major_config
from major_dataset import LoadDataset
# 设置随机种子
set_seed()
# 参数设置
MAX_EPOCH = major_config.num_epoch
BATCH_SIZE = major_config.batchsize
LR = 0.01
log_interval = 10
val_interval = 1

# ============================ step 1/5 数据 ============================
# 训练数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(256, padding=4),
    # 添加随机遮挡 旋转 等
    transforms.ToTensor(),
    transforms.Normalize(major_config.norm_mean, major_config.norm_std),
])
# 验证数据预处理
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(major_config.norm_mean, major_config.norm_std),
])

# 构建MyDataset实例
train_data = LoadDataset(data_dir=major_config.train_image, transform=train_transform)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # shuffle训练时打乱样本

# 构建DataLoder
valid_data = LoadDataset(data_dir=major_config.val_image, transform=valid_transform)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================
net = major_config.model  # 对应修改模型 net = se_resnet50(num_classes=5,pretrained=True)

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.
    # incorrect=0.

    net.train()
    for i, data in enumerate(train_loader):# 获取数据

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)  # 一个batch的loss
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)  # 1 应该是返回索引的意思
        total += labels.size(0)

        # torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。
        # pytorch可以通过.numpy()和torch.from_numpy()实现tensor和numpy的ndarray类型之间的转换
        correct += (predicted == labels).squeeze().sum().numpy()  # 计算一共正确的个数



        # 多分类时 采用one vs rest策略时 假如 label = [0,0,1,2,2] 只能百对于0,1,2这三个类别分度别计算知召道回率和专准确属率。可以用来分析数据
        #incorrect += (predicted != labels).squeeze().sum().numpy()
        # Top1就是普通的Accuracy，Top5比Top1衡量标准更“严格”，
        # 具体来讲，比如一共需要分10类，每次分类器的输出结果都是10个相加为1的概率值，Top1就是这十个值中最大的那个概率值对应的分类恰好正确的频率，而Top5则是在十个概率值中从大到小排序出前五个，然后看看这前五个分类中是否存在那个正确分类，再计算频率
        # 打印训练信息
        loss_mean += loss.item()  # 计算一共的loss
        train_curve.append(loss.item())  # 训练曲线，用于显示


        if (i+1) % log_interval == 0:   # log_interval=10 表示每迭代10次，打印一次训练信息,在这里bachsize=16 迭代10次就是160张图片，即total=160
            loss_mean = loss_mean / log_interval  # 取平均loss
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            correct=correct
            total=total   # total=160
            # 保存训练信息，即写日志
            f = open("log_training.txt", 'a')  # 若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容，在原
            # 内容之后写入。可修改该模式（'w+','w','wb'等）
            f.write("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))  # 将字符串写入文件中
            f.write("\n")  # 换行
            loss_mean = 0.  # 每次需要清0

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:  # val_interval=1 表示每一个epoch打印一次验证信息

        correct_val = 0. #  正确值
        total_val = 0.  # 一共的
        loss_val = 0.  # 损失
        net.eval()  # 模型保持静止，不进行更新，从而来验证
        with torch.no_grad():  # 不保存梯度,减少内存消耗，提高运行速度
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            valid_curve.append(loss_val/valid_loader.__len__())
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, correct_val / total_val))
            f = open("log_training.txt", 'a')  # 若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容，在原
            # 内容之后写入。可修改该模式（'w+','w','wb'等）
            f.write("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, correct_val / total_val))  # 将字符串写入文件中
            f.write("\n")  # 换行




train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()



