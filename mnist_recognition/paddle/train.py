import paddle.fluid as fluid
import paddle
from paddle.fluid import layers, dygraph
import numpy as np
from net import MyLayer
from visualization import MyVisual
from paddle.dataset.mnist import train, test
from mapper import map_number
import os
lr = 0.01
batch_size = 128
epochs = 20
use_gpu = True
model_path = 'model'

train_reader = paddle.batch(
    paddle.reader.shuffle(train(), buf_size=1000),
    batch_size=batch_size
)
test_reader = paddle.batch(
    paddle.reader.shuffle(test(), buf_size=1000),
    batch_size=batch_size
)

def save_model(model, optimizer = None):
    dygraph.save_dygraph(model.state_dict(), model_path+'/model')
    if optimizer is not None:
        dygraph.save_dygraph(optimizer.state_dict(), model_path + '/model')
def load_model(model, optimizer = None):
    if not os.path.exists(model_path):
        return
    dygraph.load_dygraph(model_path+'/model')
    if optimizer is not None:
        dygraph.load_dygraph(model_path + '/model')

def loss_acc_fn(model, images, labels):
    logits = model(images)
    loss = layers.cross_entropy(logits, labels)
    acc = layers.accuracy(logits, labels)
    avg_loss = layers.mean(loss)
    return avg_loss, acc

def update_grad(optimizer, loss):
    loss.backward()
    optimizer.minimize(loss)
    optimizer.clear_gradients()

def pre_process(data):
    images = []
    labels = []
    for image, label in data:
        images.append(image)
        labels.append(label)
    images = np.reshape(images, [-1, 1, 28, 28]).astype(np.float32)
    images = dygraph.to_variable(images)
    images = layers.resize_bilinear(images, out_shape=[32,32])
    labels = np.reshape(labels, [-1, 1]).astype(np.int64)
    labels = dygraph.to_variable(labels)
    return images, labels



place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
with dygraph.guard(place):
    model = MyLayer()
    optimizer = fluid.optimizer.SGD(learning_rate=lr, parameter_list=model.parameters())

    def train(epoch, visualer = None):
        model.train()
        print('start to train...')
        for id, data in enumerate(train_reader()):
            images, labels = pre_process(data)
            loss, acc = loss_acc_fn(model, images, labels)
            float_loss = float(loss.numpy())
            float_acc = float(acc.numpy())
            if visualer is not None:
                visualer.update_train(float_loss, float_acc)
            update_grad(optimizer, loss)
        print('Train===>Epoch: {}, Loss:{}, Accuracy: {}'.format(epoch, float_loss, float_acc))
        save_model(model)

    @dygraph.no_grad
    def test(epoch, visualer):
        model.eval()
        print('start to test...')
        loss = acc = 0.
        total = 0
        for id, data in enumerate(test_reader()):
            images, labels = pre_process(data)
            batch_loss, batch_acc = loss_acc_fn(model, images, labels)
            loss += float(batch_loss.numpy())
            acc += float(batch_acc.numpy())
            total+=1

        loss /= total
        acc /= total

        if visualer is not None:
            visualer.update_eval(loss, acc)
        print("Test===>epoch: {}, Loss: {}, Accruacy: {}".format(epoch, loss, acc))

    if __name__ == '__main__':
        visualer = MyVisual()
        for epoch in range(epochs):
            train(epoch, visualer)
            test(epoch, visualer)

