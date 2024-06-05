import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import Darknet53
from head import DetectorHead

class Yolo(nn.Module):
    def __init__(self, num_classes):
        super(Yolo, self).__init__()
        
        #########
        # Backbone
        ##
        self.backbone = Darknet53()
        
        #########
        # Multi-Heads
        ##
        # Head Block Inits NOTE: Order according to paper is scale3->scale2->scale1 such that scale2 and scale1 use the upscaled intermediate blocks
        self.scale_heads = nn.ModuleList(DetectorHead([], num_classes), # scale3 from last resblock (group)
                                         DetectorHead([], num_classes), # scale2 from 2nd to last resblock (group)
                                         DetectorHead([], num_classes)  # scale1 from 3rd to last resblock (group)
                                        )
        
        # Head Upsampling Layer Inits
        self.upsample3t2 = nn.ConvTranspose2d(stride=2)
        self.upsample2t1 = nn.ConvTranspose2d(stride=2)
        
        
    def forward(self, x):
        out, intermediates = self.backbone(x)
        
        # Heads
        
        return out
    
    def yolo_out_block(self):
        pass

if __name__ == '__main__':
    yolo_test = Yolo()