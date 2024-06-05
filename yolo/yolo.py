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
        self.upsamples = nn.ModuleList(nn.ConvTranspose2d(stride=2),    # upsample from scale3 to scale2
                                       nn.ConvTranspose2d(stride=2),    # upsample from scale2 to scale1
                                       None                             # set to None so nothing is ran when zipping
                                      )
        # self.upsample3t2 = nn.ConvTranspose2d(stride=2)
        # self.upsample2t1 = nn.ConvTranspose2d(stride=2)
        
        
    def forward(self, x):
        backbone_map, intermediates = self.backbone(x)
        
        # Heads
        head_results = []   # results will be stored in order of [scale3, scale2, scale1], i.e. results from last, 2ndtolast, 3rdtolast after head detectors
        upsampled_fm = None
        for scale_head, upsample, intermediate in zip(self.scale_heads, self.upsamples, intermediates):
            if upsampled_fm != None:    # sample3 should not concat
                intermediate = torch.concat(intermediate, upsampled_fm)
            
            detect_result, l2p_int = scale_head(intermediate)  # detect_result is the final detector prediction, l2p_int is the 2 laters previous intermediate
            head_results.append(detect_result)
            
            if upsample != None:        # sample1 should not upsample
                upsampled_fm = upsample(l2p_int)
            
        return out

if __name__ == '__main__':
    yolo_test = Yolo()