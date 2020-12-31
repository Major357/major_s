import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self, num_classes, num_linear=44944):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(num_linear, 120)  # 44944，这里nn.Linear的第一个参数随输入数据大小改变而改变
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


if __name__ == "__main__":
    # 随机生成输入数据
    rgb = torch.randn(1, 3, 224, 224)
    # 定义网络
    # # num_linear的设置是为了，随着输入图片数据大小的改变，使线性层的神经元数量可以匹配成功
    # 默认输入图片数据大小为224*224
    net = LeNet(num_classes=8,num_linear=44944)
    # 前向传播
    out = net(rgb)
    print('-----'*5)
    # 打印输出大小
    print(out.shape)
    print('-----'*5)
