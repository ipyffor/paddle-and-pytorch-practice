from paddle import fluid
from paddle.fluid import layers
from reader import get_loader
from trainer import ClassifyTrainner
from sets_path import *

epochs = 200
batch_size = 128
lr = 0.001
use_gpu = True
sets_path = sets_path
model_path = './model'
from time import time

place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
## 开启动态图模式
fluid.enable_dygraph(place)

train_loader = get_loader(sets_path+'/cifar10_train.gz', batch_size, places=place)
test_loader = get_loader(sets_path+'/cifar10_test.gz', batch_size, shuffle=False, places=place)
trainer = ClassifyTrainner(model_path=model_path, lr=lr)

def train(epoch):
    for id, (images, labels) in enumerate(train_loader()):
        loss, acc = trainer.forword(images, labels)
        if (id+1) % 50 == 0:
            print("Train ===> Epoch: {}, id: {}, loss: {}, acc: {}".format(
                epoch,
                id,
                float(loss.numpy()),
                float(acc.numpy())
            ))
        trainer.train_update()

    trainer.save_model()

def evaluation():
    sloss, sacc = 0., 0.
    total = 0
    for images, labels in test_loader:
        loss, acc = trainer.forword(images, labels, is_eval=True)
        sloss += loss.numpy()
        sacc += acc.numpy()
        total += 1
    print("Evaluation ===> avg_loss: {}, avg_acc: {}".format(
        float(sloss/total),
        float(sacc/total)
    ))

def run():

    for epoch in range(epochs):
        train(epoch)
        evaluation()

if __name__ == '__main__':
    run()
