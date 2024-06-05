import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import Backbone
from head import DetectorHead

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        
        self.backbone = Backbone()
        
        # Head Inits
        self.scale1_head = DetectorHead()
        
        
    def forward(self, x):
        out = None
        
        # Heads
        # TODO: N × N × [3 ∗ (4 + 1 + num_classes)]
        
        return out
    
    def yolo_out_block(self):
        pass

if __name__ == '__main__':
    yolo_test = Yolo()