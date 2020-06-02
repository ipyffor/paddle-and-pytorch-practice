import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.ori_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride >1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride = stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()


    def forward(self, inputs):
        out = self.ori_layers(inputs)
        out += self.shortcut(inputs)
        out = F.relu(out)
        return out

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2_x = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1)
        )
        self.conv3_x = nn.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1)
        )
        self.conv4_x = nn.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1)
        )
        self.conv5_x = nn.Sequential(
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 1),
            nn.AvgPool2d(4)
        )
        self.fc6 = nn.Linear(512, 10)

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        return x

if __name__ == '__main__':
    import numpy as np
    inputs = np.random.uniform(0, 1, [32, 3, 32, 32]).astype(np.float32)
    inputs = torch.from_numpy(inputs)
    model = Resnet18()
    outputs = model(inputs)
    print(outputs.shape)