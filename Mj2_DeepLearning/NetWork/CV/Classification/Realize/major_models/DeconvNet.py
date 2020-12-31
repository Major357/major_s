import torch
import torchvision.models as models
from torch import nn


vgg16_pretrained = models.vgg16(pretrained=False)


def decoder(input_channel, output_channel, num=3):
    if num == 3:
        decoder_body = nn.Sequential(
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, output_channel, 3, padding=1))
    elif num == 2:
        decoder_body = nn.Sequential(
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, output_channel, 3, padding=1))

    return decoder_body


class VGG16_deconv(torch.nn.Module):

    def __init__(self, num_classes=8, num_linear=131072, channel=512, height=16, width=16):
        super(VGG16_deconv, self).__init__()
        self.channel = channel;
        self.height = height;
        self.width = width
        pool_list = [4, 9, 16, 23, 30]
        for index in pool_list:
            vgg16_pretrained.features[index].return_indices = True

        self.encoder1 = vgg16_pretrained.features[:4]
        self.pool1 = vgg16_pretrained.features[4]

        self.encoder2 = vgg16_pretrained.features[5:9]
        self.pool2 = vgg16_pretrained.features[9]

        self.encoder3 = vgg16_pretrained.features[10:16]
        self.pool3 = vgg16_pretrained.features[16]

        self.encoder4 = vgg16_pretrained.features[17:23]
        self.pool4 = vgg16_pretrained.features[23]

        self.encoder5 = vgg16_pretrained.features[24:30]
        self.pool5 = vgg16_pretrained.features[30]

        self.classifier = nn.Sequential(
            torch.nn.Linear(num_linear, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, num_linear),
            torch.nn.ReLU(),
        )

        self.decoder5 = decoder(512, 512)
        self.unpool5 = nn.MaxUnpool2d(2, 2)

        self.decoder4 = decoder(512, 256)
        self.unpool4 = nn.MaxUnpool2d(2, 2)

        self.decoder3 = decoder(256, 128)
        self.unpool3 = nn.MaxUnpool2d(2, 2)

        self.decoder2 = decoder(128, 64, 2)
        self.unpool2 = nn.MaxUnpool2d(2, 2)

        self.decoder1 = decoder(64, num_classes, 2)  # classes_num
        self.unpool1 = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        print('x:', x.size())
        encoder1 = self.encoder1(x);
        print('encoder1:', encoder1.size())
        output_size1 = encoder1.size();
        pool1, indices1 = self.pool1(encoder1);
        print('pool1:', pool1.size());
        print('indices1:', indices1.size())

        encoder2 = self.encoder2(pool1);
        print('encoder2:', encoder2.size())
        output_size2 = encoder2.size()
        pool2, indices2 = self.pool2(encoder2);
        print('pool2:', pool2.size());
        print('indices2:', indices2.size())

        encoder3 = self.encoder3(pool2);
        print('encoder3:', encoder3.size())
        output_size3 = encoder3.size()
        pool3, indices3 = self.pool3(encoder3);
        print('pool3:', pool3.size());
        print('indices3:', indices3.size())

        encoder4 = self.encoder4(pool3);
        print('encoder4:', encoder4.size())
        output_size4 = encoder4.size()
        pool4, indices4 = self.pool4(encoder4);
        print('pool4:', pool4.size());
        print('indices4:', indices4.size())

        encoder5 = self.encoder5(pool4);
        print('encoder5:', encoder5.size())
        output_size5 = encoder5.size()
        pool5, indices5 = self.pool5(encoder5);
        print('pool5:', pool5.size());
        print('indices5:', indices5.size())

        pool5 = pool5.view(pool5.size(0), -1);
        print('pool5:', pool5.size())
        fc = self.classifier(pool5);
        print('fc:', fc.size())
        fc = fc.reshape(1, self.channel, self.height, self.width);
        print('fc:', fc.size())

        unpool5 = self.unpool5(input=fc, indices=indices5, output_size=output_size5);
        print('unpool5:', unpool5.size())
        decoder5 = self.decoder5(unpool5);
        print('decoder5:', decoder5.size())

        unpool4 = self.unpool4(input=decoder5, indices=indices4, output_size=output_size4);
        print('unpool4:', unpool4.size())
        decoder4 = self.decoder4(unpool4);
        print('decoder4:', decoder4.size())

        unpool3 = self.unpool3(input=decoder4, indices=indices3, output_size=output_size3);
        print('unpool3:', unpool3.size())
        decoder3 = self.decoder3(unpool3);
        print('decoder3:', decoder3.size())

        unpool2 = self.unpool2(input=decoder3, indices=indices2, output_size=output_size2);
        print('unpool2:', unpool2.size())
        decoder2 = self.decoder2(unpool2);
        print('decoder2:', decoder2.size())

        unpool1 = self.unpool1(input=decoder2, indices=indices1, output_size=output_size1);
        print('unpool1:', unpool1.size())
        decoder1 = self.decoder1(unpool1);
        print('decoder1:', decoder1.size())

        return decoder1

if __name__ == "__main__":
    # 随机生成输入数据
    rgb = torch.randn(1, 3, 512, 512)
    # 定义网络
    # num_linear的设置是为了，随着输入图片数据大小的改变，使线性层的神经元数量可以匹配成功
    # channel,height,width用于第二个fc的reshape能匹配上pool5的输出shape
    # 默认输入图片数据大小为512*512
    net = VGG16_deconv(num_classes=8,num_linear=131072,channel=512,height=16,width=16)
    # 模型参数过多，固化模型参数，降低内存损耗
    net.eval()
    # 前向传播
    out = net(rgb)
    # 打印输出大小
    print('-----'*5)
    print(out.shape)
    print('-----'*5)
