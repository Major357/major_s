import os
import random
from PIL import Image
from torch.utils.data import Dataset
import major_config
random.seed(1)

# 类别对应表
dict_label = major_config.dict_label

# 主要是用来接受索引返回样本用的
class LoadDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = dict_label  # 如果改了分类目标，这里需要修改
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform


    #接受一个索引，返回一个样本 ---  img, label

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))   # 如果改了图片格式，这里需要修改

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = dict_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info

