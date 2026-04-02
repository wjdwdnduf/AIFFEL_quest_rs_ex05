import torch
import torch.nn as nn

'''
Implementation of the standard ResNet (Residual Network) architecture.
The key feature is the 'shortcut connection' which adds the input back to the output.
'''

def conv3x3(in_planes, out_planes, stride=1):
    ''' Standard 3x3 convolution with padding to maintain spatial resolution '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    ''' Used for ResNet-18 and ResNet-34 '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride) # First convolution layer
        self.bn1 = nn.BatchNorm2d(planes)               # Batch normalization for conv1
        self.conv2 = conv3x3(planes, planes)            # Second convolution layer
        self.bn2 = nn.BatchNorm2d(planes)               # Batch normalization for conv2

        self.shortcut = nn.Sequential() # Initialize identity shortcut
        # If input size/channels don't match output, use 1x1 conv to align them
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = x # Save the input as 'identity' for the shortcut connection
        out = torch.relu(self.bn1(self.conv1(x))) # Apply conv1 -> BN -> ReLU
        out = self.bn2(self.conv2(out))           # Apply conv2 -> BN
        out += self.shortcut(identity)            # THE RESIDUAL STEP: Add identity to the output
        return torch.relu(out)                    # Apply final ReLU after addition

class Bottleneck(nn.Module):
    ''' Used for ResNet-50, 101, 152. Uses 1x1 convs to reduce/restore dimensions '''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False) # 1x1 reduction
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 3x3 spatial
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False) # 1x1 expansion
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(identity) # THE RESIDUAL STEP
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64) # Initial layer for CIFAR-10 (3 color channels)
        self.bn1 = nn.BatchNorm2d(64)
        # Create 4 stages of layers with increasing channel depths
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes) # Final fully connected layer

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) # Only the first block in a stage uses stride
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4) # Global Average Pooling
        out = out.view(out.size(0), -1) # Flatten for linear layer
        return self.linear(out)

def ResNet34(): return ResNet(BasicBlock, [3, 4, 6, 3])
def ResNet50(): return ResNet(Bottleneck, [3, 4, 6, 3])