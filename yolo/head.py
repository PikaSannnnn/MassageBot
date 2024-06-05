import torch
import torch.nn as nn
import torch.nn.functional as F

##### Main Detector Head Section
class DetectorHead(nn.Module):
    '''
    Primary head given an input feature map. The last block's intermediate layer is saved and returned for upscaling for the primary heads of other scales, if needed.
    A final Conv2d is applied and passed through a linear activation layer to produce `num_classes` output predictions.
    
    channel_sizes: [input_channel size, mid_channel size, output_channel size], e.g. [256, 128, 256]
    
    Default forward:
    intermediate (pass to next scale) <-|
                                        |
    input -> ConvHead -> ConvHead -> ConvHead -> Conv2d -> linear
    '''
    def __init__(self, channel_sizes, num_classes, num_blocks=3):
        super(DetectorHead, self).__init__()
        assert isinstance(channel_sizes, list) and len(channel_sizes) == 3, 'channel_sizes expected a list of 3 integers: in_channel, mid_channel, out_channel'
        
        self.subblocks = nn.ModuleList([ConvHead(channel_sizes) for _ in num_blocks - 1])
        self.finalblock = ConvHead(channel_sizes)   # This block will save the intermediate to upscale for other head
        self.out_layer = nn.Conv2d(channel_sizes[2], 3 * (4 + 1 + num_classes), kernel_size=1, stride=1, padding=1)
        
        self.linear_act = nn.Linear(3 * (4 + 1 + num_classes), num_classes)
        
    def forward(self, x):
        out = x.clone()
        for block in self.subblocks:
            out, _ = block(out)
            
        out, intermediate = self.finalblock(out)
        out = self.out_layer(out)
        out = self.linear_act(out)

        return out, intermediate

###### Head Block Components
class ConvHead(nn.Module):
    '''
    A subhead of 2 Conv2d layers with leaky ReLU activation functions. An intermediate feature map is saved and returned for any use.
    '''
    def __init__(self, channel_sizes):
        super(DetectorHead, self).__init__()
        assert isinstance(channel_sizes, list) and len(channel_sizes) == 3, 'channel_sizes expected a list of 3 integers: in_channel, mid_channel, out_channel'
        
        self.conv1 = nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, stride=1, padding=1)
        
        self.batch_norm1 = nn.BatchNorm2d(channel_sizes[0])
        self.batch_norm2 = nn.BatchNorm2d(channel_sizes[1])
        
    def forward(self, x):
        out = self.batch_norm1(x)
        out = self.conv1(out)
        intermediate = F.leaky_relu(out)    # Save intermediate feature map to return in case it'll be used for the upscale between scales
        
        out = self.batch_norm2(intermediate)
        out = self.conv2(out)
        out = F.leaky_relu(out)
        return out, intermediate
