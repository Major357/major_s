import random
import numpy as np
import torch
import torchvision.transforms as transforms
import os
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def caculate_mean_std():
    train_dir = os.path.join('.', "train_test")

    train_transform = transforms.Compose([

        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_data = MyDataset(data_dir=train_dir, transform=train_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=1000, shuffle=True)
    train = iter(train_loader).next()[0]  # 500张图片的mean std
    train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
    train_std = np.std(train.numpy(), axis=(0, 2, 3))

    print(train_mean, train_std)
