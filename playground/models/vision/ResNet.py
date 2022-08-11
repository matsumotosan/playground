"""ResNet implementation in PyTorch."""
import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "ResNet",
    "_resnet18",
    "_resnet18_cifar10"
    "_resnet34",
    "_resnet50",
    "_resnet101",
    "_resnet152"
]


def conv3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """Return 3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        groups=groups,
        dilation=dilation
    )


def conv1(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """Return 1x1 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        dilation=dilation
    )


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3(out_channels, out_channels, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1(out_channels, self.expansion * out_channels, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: list[int],
        in_channels: int,
        num_classes: int
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Create ResNet layers
        self.layer1 = self._make_layer(block, layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        # Set downsample 
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # Append first residual block
        layers = []
        layers.append(Block(self.in_channels, out_channels, downsample, stride))
        
        # Update in_channels
        self.in_channels = out_channels * block.expansion
        
        # Add residual blocks
        for _ in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) 
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


def _resnet18(in_channels=3, num_classes=1000):
    """Returns ResNet-18."""
    return ResNet(Block, [2, 2, 2, 2], in_channels, num_classes)


def _resnet34(in_channels=3, num_classes=1000):
    """Returns ResNet-34."""
    return ResNet(Block, [3, 4, 6, 3], in_channels, num_classes)


def _resnet50(in_channels=3, num_classes=1000):
    """Returns ResNet-50."""
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels, num_classes)


def _resnet101(in_channels=3, num_classes=1000):
    """Returns ResNet-101."""
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channels, num_classes)


def _resnet152(in_channels=3, num_classes=1000):
    """Returns ResNet-152."""
    return ResNet(Bottleneck, [3, 8, 36, 3], in_channels, num_classes)


def _resnet18_cifar10():
    """Returns ResNet-18 modified for CIFAR10 as described in SimCLR."""
    pass