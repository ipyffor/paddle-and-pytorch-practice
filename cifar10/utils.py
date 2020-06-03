import sys
import time
import math
import os, pickle, gzip

def save_object_to_zip(objects, filename):
    # if not os.path.exists(filename):
    #     file_path = os.path.split(filename)[0]
    #     if file_path and not os.path.exists(file_path):  # 需要文件夹
    #         os.mkdir(os.path.split(filename)[0])  # 创建文件夹
    #     os.mknod(filename)  # 创建文件
    fil = gzip.open(filename, 'wb')
    pickle.dump(objects, fil)
    fil.close()


def load_object_from_zip(filename):
    fil = gzip.open(filename, 'rb')
    while True:
        try:
            return pickle.load(fil)
        except EOFError:
            break
    fil.close()


class MyProcessBar():
    def __init__(self, total, process_len = 30):
        self.total = total
        self.process_len = process_len
        self.count = 0
        self.step = math.ceil(self.total/process_len)
        self.bar = ''
    def update(self, msg = ''):
        # print(msg)
        if msg != '':
            msg = '   ' + '{}'.format(msg)
        self.count += 1
        if self.count % self.step == 0 or self.count == self.total:
            self.bar+='='
            sys.stdout.write('\r|{p1:{p2}}|{p3:6.2f}%{p4}\n'.format(p1=self.bar+'>>', p2 = self.process_len+5, p3=self.count/self.total*100, p4 = msg))
            sys.stdout.flush()


import numpy as np
class SetsProcessor:

    def __init__(self, path, total, step=1):#数据范围，例如可以在索引偶数范围内，则step设置为2，间隔4个一取则设置为4
        [self.ori_data,self.enc_data, self.label] = load_object_from_zip(path)
        self.__total = total
        if total % step !=0:
            raise Exception('total % step !=0')
        self.step = step
        self.process_func = self.empty_func

    def set_process_func(self, ff):
        self.process_func = ff
    def set_step(self, step):
        self.step = step
    def batch_data(self, batch_size):

        rand = np.random.randint(int(self.__total/self.step), size=(batch_size))*self.step
        ori_data = np.zeros(shape=[batch_size, 1, 128, 128], dtype=np.float32)
        enc_data = np.zeros(shape=[batch_size, 1, 128, 128], dtype=np.float32)
        label = np.zeros(shape=[batch_size, 1], dtype=np.int64)
        for t in range(batch_size):
            ori_data[t] = np.resize(self.ori_data[rand[t]], [1, 128, 128]).astype(np.float32)
            enc_data[t] = np.resize(self.enc_data[rand[t]], [1, 128, 128]).astype(np.float32)
            label[t] = self.label[rand[t]]
        ori_data = self.process_func(ori_data)
        enc_data  =self.process_func(enc_data)
        label  =self.process_func(label)
        return ori_data/255, enc_data/255, label
    def normalization(self, x):
        minn = x.min()
        maxn = x.max()
        return (x-minn)/(maxn-minn)
    def empty_func(self,a):
        return a