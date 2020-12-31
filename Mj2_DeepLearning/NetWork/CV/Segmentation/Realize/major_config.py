from major_models import FCN,SegNet
import torch
# 1.batchsize：批次大小
batchsize = 2

# 2.num_epoch：训练轮次，一般默认200
num_epoch = 2

# 3.num_classes:分类数
num_classes = 6

# 4.crop_size:裁剪尺寸
crop_size = (512, 512)  #  (512,512)

# 5.训练集的图片和label路径
train_image = r"./major_dataset_repo/major_collected_dataset/train/image"  # r'./major_dataset_repo/major_collected_dataset/train/image'
train_label = r'./major_dataset_repo/major_collected_dataset/train/label'

# 6.验证集的图片和label路径
val_image = r'./major_dataset_repo/major_collected_dataset/validation/image'
val_label = r'./major_dataset_repo/major_collected_dataset/validation/label'

# 7.测试集的图片和label路径
test_image = r'./major_dataset_repo/major_collected_dataset/test/image'
test_label = r'./major_dataset_repo/major_collected_dataset/test/label'

# 8.待转训练、验证和测试集的数据原文件
dataset_image = r'./major_dataset_repo/major_collected_dataset/image'
dataset_label = r'./major_dataset_repo/major_collected_dataset/label'

# 9.path_test_model : 测试模型的路径
path_test_model = "./major_saved_models_repo/FCN/weights/best_model.pth"

# 10.path_predict_model : 成像模型的路径
path_predict_model = "./major_saved_models_repo/FCN/weights/best_model.pth"

# 11.模型的保存路径
path_saved_model = './major_saved_models_repo/FCN/weights/best_model.pth'

# 12.color2class_table:颜色值与类别值的对应表
path_color2class_table = "./major_dataset_repo/major_collected_dataset/color2class_table.csv"

# 13.指定设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 14.（norm_mean，norm_std）：数据集的均值和标准差
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

#15.model:模型的选择
model = SegNet.VGG16_deconv(num_classes)