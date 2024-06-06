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
    
    input -> ConvHead -> ConvHead -> ConvHead -> Conv2d -> linear
    '''
    def __init__(self, block_channels: list, num_classes: int, anchor_boxes: torch.tensor):
        super().__init__()
        # super(DetectorHead, self).__init__()
        # assert isinstance(channel_sizes, list) and len(channel_sizes) == 3, 'channel_sizes expected a list of 3 integers: in_channel, mid_channel, out_channel'
        assert isinstance(anchor_boxes, torch.Tensor), 'anchor_boxes must be a tensor of anchor box dimensions [w, l]'
        
        # Save anchor box info
        self.anchor_boxes = anchor_boxes
        self.num_anchors = self.anchor_boxes.shape[0]
        
        self.detector_blocks = nn.ModuleList([ConvHead(channel_sizes) for channel_sizes in block_channels])
        # self.subblocks = nn.ModuleList([ConvHead(channel_sizes) for _ in range(num_blocks - 1)])
        # self.finalblock = ConvHead(channel_sizes)   # This block will save the intermediate to upscale for other head
        self.out_layer = nn.Conv2d(block_channels[-1][2], self.num_anchors * (4 + 1 + num_classes), kernel_size=1, stride=1)
        
        # self.linear_act = nn.Linear(self.num_anchors * (4 + 1 + num_classes), num_classes)
        
    def forward(self, x):
        out = x.clone()
        for block in self.detector_blocks:
            out, intermediate = block(out)  # Note: most recent intermediate will be 2nd to last layer's output, which will be passed up to next scale
            
        # out, intermediate = self.detector_blocks[-1](out)
        out = self.out_layer(out)
        # out = self.linear_act(out)

        return out, intermediate

###### Head Block Components
class ConvHead(nn.Module):
    '''
    A subhead of 2 Conv2d layers with leaky ReLU activation functions. An intermediate feature map is saved and returned for any use.
    '''
    def __init__(self, channel_sizes):
        super().__init__()
        # super(DetectorHead, self).__init__()
        assert isinstance(channel_sizes, list) and len(channel_sizes) == 3, 'channel_sizes expected a list of 3 integers: in_channel, mid_channel, out_channel'
        
        self.conv1 = nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=1, stride=1, padding=0)   # Padding should be 0 as opposed to the cfg's padding=1 to preserve dimensions (padding of 1 adds 12)
        self.conv2 = nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, stride=1, padding=1)
        
        self.batch_norm1 = nn.BatchNorm2d(channel_sizes[1])
        self.batch_norm2 = nn.BatchNorm2d(channel_sizes[2])
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        intermediate = F.leaky_relu(out)    # Save intermediate feature map to return in case it'll be used for the upscale between scales
        
        out = self.conv2(intermediate)
        out = self.batch_norm2(out)
        out = F.leaky_relu(out)
        return out, intermediate
