import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import Backbone
from head import Head

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        
        self.backbone = Backbone()
        self.head = Head()
        
    def forward(self, x):
        out = None
        return out

if __name__ == '__main__':
    yolo_test = Yolo()