import pandas as pd
import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import major_config
'''np.array全部数据输出打印'''
# import numpy as np
# np.set_printoptions(threshold=np.inf)
# threshold表示: Total number of array elements to be print(输出数组的元素数目)
def access_raw_label(frame):
    '''
           读取color2class_table，将图片的rgb三通道彩色值转为一通道的class值
    '''
    #  读取color2class_table的颜色值与类别值的对应表
    dataframe = pd.read_csv(major_config.path_color2class_table)
    list_rgb = []
    list_class_id = []
    for i in range(len(dataframe)):
        rgb = str(list(dataframe.iloc[i][2:]))
        class_id = dataframe.iloc[i][0]
        list_rgb.append(rgb)
        list_class_id.append(class_id)
    dict_color2class = dict(zip(list_rgb, list_class_id))
    # 创建空数组用于存放一通道的label
    label = np.empty([major_config.crop_size[0], major_config.crop_size[1]], dtype=int)
    # print(frame.shape)  # shape内包含三个元素：按顺序为高、宽、通道数
    height = frame.shape[0]
    weight = frame.shape[1]
    #  print("weight : %s, height : %s" % (weight, height))
    # 遍历dict_color2class进行三通道与一通道的转换
    for row in range(height):            #遍历高
        for col in range(weight):         #遍历宽
            channel_values = frame[row, col]
            #  print(channel_values)
            for i in dict_color2class:
                #  print(i)
                if i == str(list(channel_values)):
                    #print("true")
                    label[row, col] = dict_color2class[i]
                    break;

    return label


class LoadDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size = crop_size

    def __getitem__(self, index):
        # 因为对image和label的路径做了排序，所以这里同一个index，就能对应上image和label
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序，这里进行了rgb转换
        label = cv2.imread(label)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序，这里进行了rgb转换
        #img, label = self.center_crop(img, label, self.crop_size) # 中心裁剪
        img, label = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'img': img, 'label': label}
        # arr_img =  img.numpy()
        # arr_label = label.numpy()
        # print("arr_img:::::",arr_img)
        # print("arr_label::::",arr_label)
        # print('处理后的图片和标签大小：', img.shape, label.shape)
        ''' **重要查看处**  '''
        # print(set(list(label.view(1, -1).unsqueeze(0)[0][0].numpy())))
        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):  # 图片的完整路径
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()  # 图片路径排序
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    # 重要修改处
    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        # 1.img:图片处理
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(major_config.norm_mean, major_config.norm_std)
            ]
        )
        img = transform_img(img)


        #  2.label:标签处理
        #  label = np.array(label)  # 以免不是np格式的数据
        label = access_raw_label(label)  # 3通道转1通道，并且进行class_id的转换
        label = torch.from_numpy(label)  # np.array转tensor
        label = label.long()  # 数据类型转long类型


        return img, label




if __name__ == "__main__":
    train = LoadDataset([major_config.train_image, major_config.train_label], major_config.crop_size)
    val = LoadDataset([major_config.val_image, major_config.val_label], major_config.crop_size)
    if major_config.test_image is not None:
        test = LoadDataset([major_config.test_image, major_config.test_label], major_config.crop_size)

