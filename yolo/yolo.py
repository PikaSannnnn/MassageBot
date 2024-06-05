import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import Backbone
from head import DetectorHead

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        
        self.backbone = Backbone()
        
        #########
        # Multi-Heads
        ##
        
        # Head Block Inits NOTE: Order according to paper is scale3->scale2->scale1 such that scale2 and scale1 use the upscaled intermediate blocks
        self.scale1_head = DetectorHead()
        self.scale2_head = DetectorHead()
        self.scale3_head = DetectorHead()
        
        # Head Upsampling Layer Inits
        self.upsample3t2 = nn.ConvTranspose2d(stride=2)
        self.upsample2t1 = nn.ConvTranspose2d(stride=2)
        
        
    def forward(self, x):
        out = None
        
        # Heads
        # TODO: N × N × [3 ∗ (4 + 1 + num_classes)]
        
        return out
    
    def yolo_out_block(self):
        pass

if __name__ == '__main__':
    yolo_test = Yolo()