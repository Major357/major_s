import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from major_evalution import eval_semantic_segmentation
from major_dataset import LoadDataset
from major_models import FCN
import major_config

# 批次大小
BATCH_SIZE = 1
# 初始化指标
miou_list = [0]
fwiou_list =[0]
# 数据读取
Load_test = LoadDataset([major_config.test_image, major_config.test_label], major_config.crop_size)
test_data = DataLoader(Load_test, batch_size=BATCH_SIZE, shuffle=True)
# 模型读入
net = major_config.model
net.eval()
net.to(major_config.device)
net.load_state_dict(torch.load(major_config.path_test_model))

# 指标初始化
train_pa = 0
train_mpa = 0
train_miou = 0
train_fwiou = 0

error = 0

for i, sample in enumerate(test_data):
        data = sample['img'].to(major_config.device)
        label = sample['label'].to(major_config.device)
        out = net(data)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = label.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        train_pa = eval_metrix['pa'] + train_pa
        train_mpa = eval_metrix['mpa'] + train_mpa
        train_miou = eval_metrix['miou'] + train_miou
        train_fwiou = eval_metrix['fwiou'] + train_fwiou

# 定义打印格式
epoch_str = ('test_pa :{:.5f} ,test_mpa:{:.5f}, test_miou:{:.5f}, test_fwiou :{:}'.format(train_pa /len(test_data),
															                              train_mpa/len(test_data),
																						  train_miou/len(test_data),
															                              train_fwiou/len(test_data)
																						  ))
# if train_miou/(len(test_data)) > max(miou_list):
# 	miou_list.append(train_miou/len(test_data))
# 	print(epoch_str+'==========last')

if train_fwiou/(len(test_data)) > max(fwiou_list):
        fwiou_list.append(train_fwiou/len(test_data))
        print(epoch_str)
