from paddle import fluid
from paddle.fluid import layers, dygraph


class ResidualBlock(dygraph.Layer):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = dygraph.Sequential(
            dygraph.Conv2D(inchannel, outchannel, filter_size=3, stride=stride, padding=1),
            dygraph.BatchNorm(outchannel, act='relu'),
            dygraph.Conv2D(outchannel, outchannel, filter_size=3, stride=1, padding=1),
            dygraph.BatchNorm(outchannel)
        )
        self.shortcut = dygraph.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = dygraph.Sequential(
                dygraph.Conv2D(inchannel, outchannel, filter_size=1, stride=stride),
                dygraph.BatchNorm(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = layers.relu(out)
        return out

class ResNet(dygraph.Layer):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = dygraph.Sequential(
            dygraph.Conv2D(3, 64, filter_size=3, stride=1, padding=1),
            dygraph.BatchNorm(64, 'relu'),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = dygraph.Linear(512, num_classes, act='softmax')

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return dygraph.Sequential(*layers)

    def forward(self, x):
        # print('input', x.min(), x.max())
        out = self.conv1(x)
        # print('c1', out.min(), out.max())
        # print(out.shape)
        out = self.layer1(out)
        # print('l1', out.min(), out.max())
        out = self.layer2(out)
        # print('l2', out.min(), out.max())
        out = self.layer3(out)
        # print('l3', out.min(), out.max())
        out = self.layer4(out)
        # print('l4', out.min(), out.max())
        out = layers.pool2d(out, 4)
        out = layers.flatten(out)
        out = self.fc(out)
        # print('fc', out.min(), out.max())
        return out


def ResNet18():

    return ResNet(ResidualBlock)
