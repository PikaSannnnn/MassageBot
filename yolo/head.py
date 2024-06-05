import torch
import torch.nn as nn
import torch.nn.functional as F

##### Main Detector Head Section
class DetectorHead(nn.Module):
    def __init__(self, channel_sizes, num_blocks=3):
        super(DetectorHead, self).__init__()
        assert isinstance(channel_sizes, list) and len(channel_sizes) == 3, 'channel_sizes expected a list of 3 integers: in_channel, mid_channel, out_channel'
        
        self.subblocks = nn.ModuleList([ConvHead(channel_sizes) for _ in num_blocks - 1])
        self.finalblock = ConvHead(channel_sizes)
        self.out_layer = nn.Conv2d(channel_sizes[2], 255, kernel_size=1, stride=1, padding=1)
        
    def forward(self, x):
        out = x.clone()
        for block in self.subblocks:
            out, _ = block(out)
            
        out, intermediate = self.finalblock(out)
        out = self.out_layer(out)

        return out, intermediate

###### Head Block Components
class ConvHead(nn.Module):
    def __init__(self, channel_sizes):
        super(DetectorHead, self).__init__()
        assert isinstance(channel_sizes, list) and len(channel_sizes) == 3, 'channel_sizes expected a list of 3 integers: in_channel, mid_channel, out_channel'
        
        self.conv1 = nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        intermediate = F.leaky_relu(out)
        
        out = self.conv2(intermediate)
        out = F.leaky_relu(out)
        return out, intermediate
