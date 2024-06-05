import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.vovnet = None  # torchvision.models.vgg16(pretrained=True)
        
    def forward(self, x):
        out = None
        return out
    
    def init_params(self, weights):
        assert len(weights) == 4, 'weights must contain the pre-trained weights for the 4 backbone layers. Use None if no pre-trained weights are used'
        
        for pt_net, net_name in zip(weights, ['VoVNET', 'CSPVoVNET', 'ELAN', 'E-LAN']):
            pass
    
class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
    
    def forward(self, x):
        out = None
        return out
    
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
    
    def forward(self, x):
        out = None
        return out

class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ConvResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, 3, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(in_channels)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = x + out
        out = self.leaky_relu(out)
        return out
    
class Darknet53(nn.Module):
    # Darknet53 input should be 416x416x3 or 608x608x3
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False) # 256x256x3 -> 256x256x32
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False) # 256x256x32 -> 128x128x64
        self.bn2 = nn.BatchNorm2d(64)
        self.residual_block1 = ConvResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False) # 128x128x64 -> 64x64x128
        self.bn3 = nn.BatchNorm2d(128)
        self.residual_block2 = self._make_layer(ConvResidualBlock, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1, bias=False) # 64x64x128 -> 32x32x256
        self.bn4 = nn.BatchNorm2d(256)
        self.residual_block3 = self._make_layer(ConvResidualBlock, 256, 8)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1, bias=False) # 32x32x256 -> 16x16x512
        self.bn5 = nn.BatchNorm2d(512)
        self.residual_block4 = self._make_layer(ConvResidualBlock, 512, 8) # 16x16x512 -> 16x16x512
        self.conv6 = nn.Conv2d(512, 1024, 3, 2, 1, bias=False) # 16x16x512 -> 8x8x1024
        self.bn6 = nn.BatchNorm2d(1024)
        self.residual_block5 = self._make_layer(ConvResidualBlock, 1024, 4) # 8x8x1024 -> 8x8x1024
        self.avgpool = nn.AdaptiveAvgPool2d(1) # 8x8x1024 -> 1x1x1024
        self.fc = nn.Linear(1024, 1000)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x) # 256x256x3 -> 256x256x32
        out = self.bn1(out) # 256x256x32 -> 256x256x32
        out = self.leaky_relu(out)
        out = self.conv2(out) # 256x256x32 -> 128x128x64
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.residual_block1(out) # 128x128x64 -> 128x128x64
        out = self.conv3(out) # 128x128x64 -> 64x64x128
        out = self.bn3(out)
        out = self.leaky_relu(out)
        out = self.residual_block2(out)
        out = self.conv4(out) # 64x64x128 -> 32x32x256
        out = self.bn4(out)
        out = self.leaky_relu(out)
        out = self.residual_block3(out)
        out = self.conv5(out) # 32x32x256 -> 16x16x512
        out = self.bn5(out)
        out = self.leaky_relu(out)
        out = self.residual_block4(out) #1
        out = self.conv6(out) # 16x16x512 -> 8x8x1024
        out = self.bn6(out)
        out = self.leaky_relu(out)
        out = self.residual_block5(out) #1
        out = self.avgpool(out) # 8x8x1024 -> 1x1x1024
        out = torch.flatten(out,1) # 1x1x1024 -> 1024
        out = self.fc(out) # 1024 -> 1000
        out = self.softmax(out) # 1000
        return out
