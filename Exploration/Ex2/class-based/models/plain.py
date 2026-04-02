import torch
import torch.nn as nn
from .resnet import conv3x3, ResNet # Inherit the main ResNet structure

'''
Implementation of Plain Networks for Ablation Study.
This model intentionally removes the identity addition to demonstrate its impact.
'''

class PlainBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(PlainBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # Note: No shortcut parameter is needed here as we won't use it.

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # TASK 3: ABLATION STUDY - Identity addition is removed to simulate a 'plain' network.
        return torch.relu(out)

class PlainBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(PlainBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # TASK 3: ABLATION STUDY - Residual connection removed.
        return torch.relu(out)

def PlainNet34(): return ResNet(PlainBasicBlock, [3, 4, 6, 3])
def PlainNet50(): return ResNet(PlainBottleneck, [3, 4, 6, 3])