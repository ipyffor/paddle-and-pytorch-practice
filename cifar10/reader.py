import sys
sys.path.append('..')
from utils import load_object_from_zip
import paddle.fluid as fluid
import numpy as np
import cv2
# import paddle.dataset.cifar as cifar
# cifar.train10()
class MyDataset(fluid.io.Dataset):
    def __init__(self, path, transform = None, target_transform = None):
        super(MyDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.images, self.labels = load_object_from_zip(path)


    def __len__(self):
        len_images = len(self.images)
        assert len_images == len(self.labels)
        return len_images
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        image = image.transpose([2, 0, 1])
        return image, label


def get_loader(path, batch_size, shuffle = True, transform = None, target_transform = None,num_workers=0, places = None):
    dataset = MyDataset(path,transform = transform, target_transform = target_transform)
    # loader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)
    # loader.set_sample_list_generator(dataset, places=places)
    loader = fluid.io.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, return_list=True,places=places)
    return loader
# def get_reader():
#     reader =

if __name__ == '__main__':
    fluid.enable_dygraph(fluid.CPUPlace())
    loader = get_loader('../data_sets/cifar10_train.gz', 20, places=fluid.CPUPlace())
    for data,label in loader:
        print(label.shape)
        data = data.numpy()*255
        data = data.astype(np.uint8)
        data = np.transpose(data, (0,2,3,1))
        cv2.imshow('img', data[0])
        cv2.waitKey()
        break
