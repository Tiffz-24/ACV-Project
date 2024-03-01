from torch import nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding = 1)
        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
          #Use a Conv2d layer with kernel_size=1 to "resize" input
          self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity_x = self.skip_connection(x)
        identity_x = self.conv1(identity_x)
        identity_x = F.relu(identity_x)
        identity_x = self.conv2(identity_x)
        x = x + identity_x
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Need to crop the input to focus on the central 32x32 region
        # start off with 64 3x32x32 images
        # need to adjust the numbers for this dataset
        self.conv1 = nn.Conv2d(3, 32, 9) # 64x32x24x24
        self.pool1 = nn.MaxPool2d(2) # 64x32x12x12
        self.ResNet1 = ResNetBlock(32, 64, stride=2) # 64x64x12x12
        self.ResNet2 = ResNetBlock(64, 128) # 64x128x12x12

        self.conv2 = nn.Conv2d(128, 256, 5) # 64x256x8x8
        self.conv3 = nn.Conv2d(256, 256, 5) # 64x256x4x4
        self.pool2 = nn.MaxPool2d(2) # 64x256x2x2

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 100)
        self.lin3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.ResNet1(x)
        x = self.ResNet2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x

