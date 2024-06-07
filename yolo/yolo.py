import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Darknet53
from .head import DetectorHead

class Yolo(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()
        # super(Yolo, self).__init__()
#         assert isinstance(anchor_boxes, torch.Tensor), 'anchor_boxes must be a tensor of anchor box dimensions [w, l]'
        
#         self.anchor_boxes = anchor_boxes
        self.num_anchors = num_anchors
        
        #########
        # Backbone
        ##
        self.backbone = Darknet53()
        
        #########
        # Multi-Heads
        ##
        # Head Block Inits NOTE: Order according to paper is scale3->scale2->scale1 such that scale2 and scale1 use the upscaled intermediate blocks
        self.scale_heads = nn.ModuleList([DetectorHead([[1024, 512, 1024]] * 3, num_classes, self.num_anchors),                        # scale3 from last resblock (group)
                                          DetectorHead([[1024, 256, 512]] + [[512, 256, 512]] * 2, num_classes, self.num_anchors),    # scale2 from 2nd to last resblock (group)
                                          DetectorHead([[512, 128, 256]] + [[256, 128, 256]] * 2, num_classes, self.num_anchors)      # scale1 from 3rd to last resblock (group)
                                        ])
        
        # Head Upsampling Layer Inits
        self.upsamples = nn.ModuleList([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),      # upsample from scale3 to scale2
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),      # upsample from scale2 to scale1
                                        None         # set to None so nothing is ran when zipping
                                       ])
        # self.upsample3t2 = nn.ConvTranspose2d(stride=2)
        # self.upsample2t1 = nn.ConvTranspose2d(stride=2)
        
        
    def forward(self, x):
        print('begin yolo')
        intermediates = self.backbone(x)
        print('backbone end return')
        
        # Heads
        head_results = []   # results will be stored in order of [scale3, scale2, scale1], i.e. results from last, 2ndtolast, 3rdtolast after head detectors
        upsampled_fm = None
        [print(i.shape) for i in intermediates]
        for scale_head, upsample, intermediate in zip(self.scale_heads, self.upsamples, intermediates):
            print(intermediate.shape)
            if upsampled_fm != None:    # sample3 should not concat
                print(intermediate.shape, upsampled_fm.shape)
                intermediate = torch.concat((intermediate, upsampled_fm), dim=1)
                # int1 = [10, 1024, 20, 20]
                    # [10, 512, 20, 20] -> [10, 256, 20, 20] -> [10, 256, 40, 40]
                # int2 = [10, 512, 40, 40], upsampled = [10, 512, 40, 40] => [10, 1024, 40, 40]
                # int3 = [10, 256, 80, 80], upsampled = [10, 512, 80, 80] => [10, 768, 80, 80]
            
            detect_result, l2p_int = scale_head(intermediate)  # detect_result is the final detector prediction, l2p_int is the 2 laters previous intermediate
            head_results.append(detect_result)
            print('l2p', l2p_int.shape)
            
            if upsample != None:        # sample1 should not upsample
                upsampled_fm = upsample(l2p_int)
            
        return head_results

if __name__ == '__main__':
    yolo_test = Yolo()