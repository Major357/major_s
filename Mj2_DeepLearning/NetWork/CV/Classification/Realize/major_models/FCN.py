import numpy as np
import torch
from torchvision import models
from torch import nn


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight)


pretrained_net = models.vgg16_bn(pretrained=False)


class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stage1 = pretrained_net.features[:7]
        self.stage2 = pretrained_net.features[7:14]
        self.stage3 = pretrained_net.features[14:24]
        self.stage4 = pretrained_net.features[24:34]
        self.stage5 = pretrained_net.features[34:]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.upsample_2x_1.weight.data = bilinear_kernel(512, 512, 4)

        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.upsample_2x_2.weight.data = bilinear_kernel(256, 256, 4)

    def forward(self, x):
        # print('image:', x.size())

        s1 = self.stage1(x)
        # print('pool1:', s1.size())

        s2 = self.stage2(s1)
        # print('pool2:', s2.size())

        s3 = self.stage3(s2)
        # print('pool3:', s3.size())

        s4 = self.stage4(s3)
        # print('pool4:', s4.size())

        s5 = self.stage5(s4)
        # print('pool5:', s5.size())

        scores1 = self.scores1(s5)  # self.scores1 = nn.Conv2d(512, num_classes, 1); 这里进行了一次通道数的变化
        # print('scores1:', scores1.size())

        s5 = self.upsample_2x_1(s5)  # nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False); 转置卷积进行第一次上采样
        # print('s5:', s5.size())

        ##############融合##################
        add1 = s5 + s4  # 第一次上采样 与 s4进行融合
        # print('add1:', add1.size())

        scores2 = self.scores2(add1)  # self.scores2 = nn.Conv2d(512, num_classes, 1)  将融合后的add1进行一次通道数变化为num_classes
        # print('scores2:', scores2.size())

        add1 = self.conv_trans1(add1)  # self.conv_trans1 = nn.Conv2d(512, 256, 1) 将融合后的add1进行一次通道数变化为256
        # print('add1:', add1.size())

        add1 = self.upsample_2x_2(
            add1)  # self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False) 将通道256的add1 ,上采样为add1
        # print('add1:', add1.size())

        add2 = add1 + s3  # 将add1  和 s3 进行融合
        # print('add2:', add2.size())

        output = self.conv_trans2(add2)  # self.conv_trans2 = nn.Conv2d(256, num_classes, 1) 改变add2的通道数
        # print('output:', output.size())

        output = self.upsample_8x(
            output)  # self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        # 使用转置卷积进行上采样
        # print('output:', output.size())

        return output


if __name__ == "__main__":
    # 随机生成输入数据
    rgb = torch.randn(1, 3, 360, 480)
    # 定义网络
    net = FCN(12)
    # 前向传播
    out = net(rgb)
    # 打印输出大小
    print('-----' * 5)
    print(out.shape)
    print('-----' * 5)
