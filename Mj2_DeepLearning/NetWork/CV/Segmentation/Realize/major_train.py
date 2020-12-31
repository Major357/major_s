import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from major_dataset import LoadDataset
from major_evalution import eval_semantic_segmentation
import major_config

# ****************************************step1 数据处理**********************************************#
Load_train = LoadDataset([major_config.train_image, major_config.train_label], major_config.crop_size)
Load_val = LoadDataset([major_config.val_image, major_config.val_label], major_config.crop_size)

train_data = DataLoader(Load_train, batch_size=major_config.batchsize, shuffle=True, num_workers=1)
val_data = DataLoader(Load_val, batch_size=major_config.batchsize, shuffle=True, num_workers=1)

# *****************************************step2 模型*********************************************#
net = major_config.model
net = net.to(major_config.device)

# ******************************************step3 损失函数********************************************#
criterion = nn.NLLLoss().to(major_config.device)  # NLLLoss有利于最后激活层的替换

# ******************************************step4 优化器********************************************#
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# ******************************************step5 训练********************************************#
def train(model):
    best = [0]  # 存储最优指标，用于Early Stopping
    net = model.train()  # 指定模型为训练模式，即可以进行参数更新
    # 训练轮次
    for epoch in range(major_config.num_epoch):
        print('Epoch is [{}/{}]'.format(epoch + 1, major_config.num_epoch))
        # 每20次epoch,lr学习率降一半
        if epoch % 20 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        # 指标初始化
        train_loss = 0
        train_pa = 0
        train_mpa = 0
        train_miou = 0
        train_fwiou = 0
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = sample['img'].to(major_config.device)
            img_label = sample['label'].to(major_config.device)
            # 训练
            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)  # loss计算
            optimizer.zero_grad()  # 需要梯度清零，再反向传播
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            train_loss += loss.item()  # loss累加
            # 评估
            # 预测值
            pre_label = out.max(dim=1)[1].data.cpu().numpy()  # [1]：表示返回索引
            pre_label = [i for i in pre_label]
            # 真实值
            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]
            # 计算所有的评价指标
            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            # 各评价指标计算
            train_pa += eval_metrix['pa']
            train_mpa += eval_metrix['mpa']
            train_miou += eval_metrix['miou']
            train_fwiou += eval_metrix['fwiou']
            #  打印损失
            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))
        #  评价指标打印格式定义
        metric_description = '|Train PA|: {:.5f}|\n|Train MPA|: {:.5f}|\n|Train MIou|: {:.5f}|\n|Train FWIou|: {:.5f}|'.format(
            train_pa / len(train_data),
            train_mpa / len(train_data),
            train_miou / len(train_data),
            train_fwiou / len(train_data),
        )
        #  打印评价指标
        print(metric_description)
        #  根据train_miou，保存最优模型
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            torch.save(net.state_dict(), major_config.path_saved_model)

# ******************************************step6 评价********************************************#
def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = sample['img'].to(major_config.device)
        valLabel = sample['label'].long().to(major_config.device)

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
        eval_loss / len(train_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)))
    print(val_str)
    print(time_str)


if __name__ == "__main__":
    train(net)
    # evaluate(net) 验证可以自己设置每训练多少次，验证一次，所以，evaluate()函数可以放到train()里面

