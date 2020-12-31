import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from major_dataset import LoadDataset
import cv2
import major_config

# 导入数据
Load_test = LoadDataset([major_config.test_image, major_config.test_label], major_config.crop_size)
test_data = DataLoader(Load_test, batch_size=1)
# 导入模型
net = major_config.model
net.eval() # 参数固化
net.to(major_config.device) # 送入指定设备
# 加载模型参数
net.load_state_dict(torch.load(major_config.path_predict_model))
# 加载color2class_table:颜色值与类别值的对应表
color2class_table = pd.read_csv(major_config.path_color2class_table)
# predict
for i, sample in enumerate(test_data):
		valImg = sample['img'].to(major_config.device)
		out = net(valImg)
		out = F.log_softmax(out, dim=1)
		pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
		print(pre_label)
		# 多图预测 batch_size>=2
		# pre_label = pre_label[0]
		cv2.imwrite(str(i)+".png",pre_label)
		#print(type(pre_label))
		#print(i)
		img_show = Image.open(str(i)+".png")
		img_show.show()



