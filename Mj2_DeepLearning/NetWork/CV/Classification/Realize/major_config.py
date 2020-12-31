from major_models.LeNet import LeNet
import torch

# 1.dict_label:类别对应表
dict_label = {"banana": 0, "cat": 1, "dog": 2, "dolphin": 3, "person": 4,"pig": 5}  # 如果改了分类目标，这里需要修改

# 2.batchsize：批次大小
batchsize = 2

# 3.num_epoch：训练轮次，一般默认200
num_epoch = 2

# 4.crop_size:裁剪尺寸
crop_size = (256, 256)  #  (512,512)

# 5.训练集的图片路径
train_image = r"./major_dataset_repo/major_collected_dataset/split_data/train"  # r'./major_dataset_repo/major_collected_dataset/train/image'

# 6.验证集的图片路径
val_image = r'./major_dataset_repo/major_collected_dataset/split_data/valid'

# 7.测试集的图片路径
test_image = r'./major_dataset_repo/major_collected_dataset/split_data/test'

# 8.待转训练、验证和测试集的数据原文件
dataset_image = r'./major_dataset_repo/major_collected_dataset/image'

# 9.path_test_model : 测试模型的路径
path_test_model = "./major_saved_models_repo/FCN/weights/best_model.pth"

# 10.path_predict_model : predict模型的路径
path_predict_model = "./major_saved_models_repo/FCN/weights/best_model.pth"

# 11.模型的保存路径
path_saved_model = './major_saved_models_repo/FCN/weights/best_model.pth'

# 12.指定设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# （norm_mean，norm_std）：数据集的均值和标准差
norm_mean = [0.33424968,0.33424437, 0.33428448]
norm_std = [0.24796878, 0.24796101, 0.24801227]

#15.model:模型的选择
model = LeNet(num_classes=6,num_linear=44944)