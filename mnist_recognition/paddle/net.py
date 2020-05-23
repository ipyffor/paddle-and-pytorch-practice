import paddle.fluid as fluid
from paddle.fluid import layers, dygraph
import numpy as np

class C1(dygraph.Layer):
    def __init__(self):
        super(C1, self).__init__()
        self.c1 = dygraph.Sequential(
            dygraph.Conv2D(num_channels=1, num_filters=6, filter_size=5, stride=1, act='relu'),
            dygraph.Pool2D(2, 'max', 2)
        )
    def forward(self, inputs):
        return self.c1(inputs)

class C2(dygraph.Layer):
    def __init__(self):
        super(C2, self).__init__()
        self.c2 = dygraph.Sequential(
            dygraph.Conv2D(6, 16, 5, 1, act='relu'),
            dygraph.Pool2D(2, 'max', 2)
        )
    def forward(self, inputs):
        return self.c2(inputs)

class C3(dygraph.Layer):
    def __init__(self):
        super(C3, self).__init__()
        self.c3 = dygraph.Sequential(
            dygraph.Conv2D(16, 120, 5, 1, act='relu')
        )
    def forward(self, inputs):
        return self.c3(inputs)

class F4(dygraph.Layer):
    def __init__(self):
        super(F4, self).__init__()
        self.f4 = dygraph.Sequential(
            dygraph.Linear(120, 84, act='relu')
        )
    def forward(self, inputs):
        return self.f4(inputs)

class F5(dygraph.Layer):
    def __init__(self):
        super(F5, self).__init__()
        self.f5 = dygraph.Sequential(
            dygraph.Linear(84, 10, act='softmax')
        )
    def forward(self, inputs):
        return self.f5(inputs)



class MyLayer(dygraph.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

        self.c1 = C1()
        self.c2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()
    def forward(self, inputs):
        x = inputs
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = layers.flatten(x)
        x = self.f4(x)
        x = self.f5(x)
        return x


if __name__ == '__main__':
    with dygraph.guard(fluid.CPUPlace()):
        model = MyLayer()
        input = np.random.uniform(-1,1, [4, 1, 32, 32]).astype(np.float32)
        input_var = dygraph.to_variable(input)
        output = model(input_var)
        print(output.shape)
