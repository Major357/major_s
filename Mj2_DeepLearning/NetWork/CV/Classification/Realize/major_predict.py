import os
from major_dataset import LoadDataset
import major_config
import torch
import torchvision.transforms as transforms



valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(major_config.norm_mean, major_config.norm_std),
])


net = major_config.model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(BASE_DIR, "test_data")

test_data = LoadDataset(data_dir=test_dir, transform=valid_transform)
valid_loader = LoadDataset(dataset=test_data, batch_size=1)

for i, data in enumerate(valid_loader):
    # forward
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1) # 取最大的索引

    #rmb = 1 if predicted.numpy()[0] == 0 else 100
    final_choose=[1,2,3,4,5]
    result=final_choose[predicted.numpy()[0]]
    print("模型获得类型{}".format(result))