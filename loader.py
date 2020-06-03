import torch
import torchvision
from torch.utils.data import DataLoader
import os


def get_loader(path,batch_size, shuffle = False, num_workers = 0,transform = None, train=True, download = False):
    dataset = torchvision.datasets.CIFAR10(path, train=train, download=download, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

if __name__ == '__main__':
    from torchvision import transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = get_loader(50000, shuffle=True, num_workers=0, transform=transform_train, train=True, download=False)
    test_data = get_loader(1, shuffle=False, num_workers=0, train=False, download=False)
    print(train_data.__len__(), test_data.__len__())
    for i, (img, label) in enumerate(train_data):
        print(img.min(), img.max())