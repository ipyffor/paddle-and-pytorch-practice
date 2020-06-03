from paddle import fluid
import math
from paddle.fluid import dygraph, layers
from paddle.fluid.layers import reduce_max as rmax, reduce_min as rmin

def compute_padding(input_size,output_size , filter_size, stride):
    padding = math.ceil(((output_size-1)*stride-(input_size-filter_size))/2.)
    return padding

class ResidualBlock(dygraph.Layer):
    def __init__(self, num_channels, num_filters,stride = 1):
        super(ResidualBlock, self).__init__()
        self.origin_layers = dygraph.Sequential(
            dygraph.Conv2D(num_channels, num_filters, 3, stride, padding=1),
            dygraph.BatchNorm(num_filters, act='relu'),
            dygraph.Conv2D(num_filters, num_filters, 3, 1, padding=1),
            dygraph.BatchNorm(num_filters)
        )
        if stride != 1:
            self.shortcut = dygraph.Sequential(
                dygraph.Conv2D(num_channels, num_filters, 3, stride, padding=1),
                dygraph.BatchNorm(num_filters)
            )
        else:
            self.shortcut = dygraph.Sequential()
    def forward(self, inputs):
        x = inputs
        out_origin_layers = self.origin_layers(x)
        out_shortcut = self.shortcut(x)
        # print("out_origin_layers", out_origin_layers.shape, "out_shortcut", out_shortcut.shape)

        out = layers.relu((out_origin_layers+out_shortcut))
        return out

class Resnet18(dygraph.Layer):
    def __init__(self):
        super(Resnet18, self).__init__()
        ##3*128*128
        self.c1 = dygraph.Sequential(
            dygraph.Conv2D(3, 64, 7, 2, padding=51),
            dygraph.BatchNorm(64, act='relu'),
            dygraph.Conv2D(64, 64, 3, 2, padding=1),
            dygraph.BatchNorm(64, act='relu'),
            # dygraph.Pool2D(3, 'max', 2, pool_padding=1)
        )
        ##64*56*56
        self.c2_x = dygraph.Sequential(
            ResidualBlock(64, 64, 2),
            ResidualBlock(64, 64, 1)
        )
        ##64*28*28
        self.c3_x = dygraph.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1)
        )
        ##128*14*14
        self.c4_x = dygraph.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1),
            ##256*7*7
        )
        self.c5_x = dygraph.Sequential(
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 1),
            dygraph.Pool2D(3,'avg', 2)
        )
        ##256*1*1
        self.f6 = dygraph.Sequential(
            dygraph.Linear(512, 10, act='softmax')
        )
        ##10
    def forward(self, inputs):
        x = inputs
        # print("inputs", rmin(x).numpy(), rmax(x).numpy())
        x = self.c1(x)
        # print("c1", rmin(x).numpy(), rmax(x).numpy())
        x =self.c2_x(x)
        # print("c2_x", rmin(x).numpy(), rmax(x).numpy())
        x = self.c3_x(x)
        x = self.c4_x(x)
        x = self.c5_x(x)
        # print(x.shape)
        # print("c5_x", rmin(x).numpy(), rmax(x).numpy())
        x = layers.flatten(x)
        x = self.f6(x)
        # print("f6", rmin(x).numpy(), rmax(x).numpy())
        return x

if __name__ == '__main__':
    import numpy as np
    from time import time
    fluid.enable_dygraph(fluid.CUDAPlace(0))
    model = Resnet18()
    inputs = np.random.uniform(0,1, [128, 3, 128, 128]).astype(np.float32)
    inputs = dygraph.to_variable(inputs)
    start = time()
    outputs = model(inputs)
    end = time()
    print('前向时间',end-start)
    print(outputs.shape)

