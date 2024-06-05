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