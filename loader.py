import torch
import torchvision
from torch.utils.data import DataLoader


def get_loader(batch_size, shuffle = False, num_workers = 0,transform = None, train=True, download = False):
    dataset = torchvision.datasets.CIFAR10('./data_sets', train=train, download=download, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

if __name__ == '__main__':
    train_data = get_loader(1, shuffle=True, num_workers=0, train=True, download=False)
    test_data = get_loader(1, shuffle=False, num_workers=0, train=False, download=False)
    print(train_data.__len__(), test_data.__len__())