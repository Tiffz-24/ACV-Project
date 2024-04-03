from torch import nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        print("RESNET BLOCK FORWARD: ", x.shape)
        identity = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Define ResNet layers
        self.layer1 = ResNetBlock(64, 64)
        self.layer2 = ResNetBlock(64, 128, stride=2)
        self.layer3 = ResNetBlock(128, 256, stride=2)
        self.layer4 = ResNetBlock(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # This makes the network input size agnostic
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        print("FORWARD: ", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # Adaptive pooling
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
