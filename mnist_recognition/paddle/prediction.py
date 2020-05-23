import paddle.fluid as fluid
import paddle
from paddle.fluid import layers, dygraph
import numpy as np
from net import MyLayer
from visualization import MyVisual
from paddle.dataset.mnist import train, test
from mapper import map_number
import sys
sys.path.append('..')
from utils import load_object_from_zip
import os

model_path = 'model'

def load_model(model, optimizer = None):
    if not os.path.exists(model_path):
        return
    model_dict, optimizer_dict = dygraph.load_dygraph(model_path+'/model')
    model.load_dict(model_dict)
    if optimizer is not None:
        optimizer.set_dict(optimizer_dict)

images,labels = load_object_from_zip('../data_sets/test0_255_complete.gzip')


@dygraph.no_grad
def pred_fn(model, images):
    images = (np.reshape(images, [-1, 1, 28, 28]).astype(np.float32)/255)*2-1
    images = dygraph.to_variable(images)
    images = layers.resize_bilinear(images, out_shape=[32,32])
    logits = model(images).numpy()
    return [map_number(logit) for logit in logits]

with dygraph.guard(fluid.CPUPlace()):
    model = MyLayer()
    model.eval()
    load_model(model)
    indexs = np.random.randint(0, 10000, size=[20])
    pred = pred_fn(model, images[indexs])
    print('{:10}{}\n{:10}{}'.format('Label: ', pred, 'Pred: ', list(labels[indexs])))
