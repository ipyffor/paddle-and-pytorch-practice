# from classify_net import Resnet18
from resnet import ResNet18
from paddle import fluid
from paddle.fluid import dygraph, layers
# import numpy as np
from time import time

class ClassifyTrainner():
    def __init__(self, model_path, lr = 0.001, train_on = True):
        self.model_path = model_path
        self.model = ResNet18()
        if train_on:
            self.optim = fluid.optimizer.MomentumOptimizer(learning_rate=lr, momentum=0.9, parameter_list=self.model.parameters(), regularization=fluid.regularizer.L2Decay(5e-4))
    def set_model_path(self,path):
        self.model_path = path
    def pre_process(self, images, labels):
        [images, labels] = [dygraph.to_variable(x) for x in (images, labels)]
        # images = layers.resize_bilinear(images, out_shape=[128, 128])
        labels = layers.reshape(labels, shape=[-1,1])
        return images, labels
    def loss_acc_fn(self, images, labels):
        images, labels = self.pre_process(images, labels)
        # print(images.shape)
        logits = self.model(images)
        self.loss = layers.mean(layers.cross_entropy(logits, labels))
        # print(layers.reduce_max(self.loss).numpy())
        self.acc = layers.accuracy(logits, labels)

    def forword(self, images, labels, is_eval = False):
        if is_eval:
            self.model.eval()
            with dygraph.no_grad():
                self.loss_acc_fn(images, labels)
            self.model.train()
            return self.loss, self.acc
        self.loss_acc_fn(images, labels)
        return self.loss, self.acc
    def train_update(self):
        self.loss.backward()
        self.optim.minimize(self.loss)
        self.optim.clear_gradients()
    def save_model(self, optim = None):
        dygraph.save_dygraph(self.model.state_dict(), self.model_path+'/resnet18')
        if optim is not None:
            dygraph.save_dygraph(self.optim.state_dict(), self.model_path+'/resnet18')
    def load_model(self, load_optim = False):
        m, o = dygraph.load_dygraph(self.model_path+'resnet18')
        self.model.load_dict(m)
        if load_optim:
            self.optim.set_dict(o)