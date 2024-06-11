import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MultiFactorLoss(nn.Module):
    def __init__(self, num_anchors, obj_weights=(1, 10)):
        super().__init__()
        self.num_anchors = num_anchors
        
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCEWithLogitsLoss()
        # self.bceloss = nn.BCELoss()
        self.celoss = nn.CrossEntropyLoss()

        assert len(obj_weights) == 2, 'obj_weights must be a 2d tuple or list of the (Obj_weight, and NoObj_weight)'
        self.obj_weights = obj_weights

    def ObjLoss(self, target_coords, preds, scale):
        OBJ_WEIGHT, NOOBJ_WEIGHT = self.obj_weights
        
        # OBJ_WEIGHT = 20
        # NOOBJ_WEIGHT = 0.1
        assert preds.shape[0] == self.num_anchors, f'prediction should have {self.num_anchors} anchors'
        
        target_coords = torch.unique(target_coords, dim=0)
        # print(target_coords.shape)
        
        loss = 0
        target_OH = torch.zeros(scale, scale)
        target_OH[target_coords[:, 0], target_coords[:, 1]] = 1
        target_OH = target_OH.view(-1)
        target_OH_noobj = 1 - target_OH
        
        target_OH = torch.ones(target_coords.shape[0])
        target_OH_noobj = torch.zeros((scale * scale) - target_coords.shape[0])
        
        # Original. Repeat
        target_OH = target_OH.repeat(self.num_anchors, 1)
        target_OH_noobj = target_OH_noobj.repeat(self.num_anchors, 1)
        
        
        # Original
        # pred_OH = preds[:, 4, :, :]
        # pred_OH_test = preds[:, 4, target_coords[:, 0], target_coords[:, 1]]
        
        pred_OH = preds.view(self.num_anchors, 19, -1) # Flatten 13x13 -> 169, etc.
        noobj_coords = torch.sum(target_coords[:] * torch.tensor([scale, 1]), dim=1) # Get flattened indices from coords
        obj_preds = pred_OH[:, 4, noobj_coords] # Get obj scores from obj cells
        
        no_obj_mask = torch.ones(scale * scale, dtype=torch.bool) # Mask to keep only the no_obj cells
        no_obj_mask[noobj_coords] = False
        # print(no_obj_mask)
        no_obj_preds = pred_OH[:, 4, no_obj_mask]
        # print(pred_OH[:, 4, no_obj_mask].shape)
        
        # print(no_obj_mask)
        # print(test_coords)
        # print(target_OH.shape, target_OH_noobj.shape, '|', obj_preds.shape, no_obj_preds.shape, '|', test_coords.shape, target_coords.shape)
        
        bce_objloss = self.bceloss(obj_preds, target_OH)
        bce_noobjloss = self.bceloss(no_obj_preds, target_OH_noobj)
        
        # print((OBJ_WEIGHT * bce_objloss) + (NOOBJ_WEIGHT * bce_no_objloss))
        
        return (OBJ_WEIGHT * bce_objloss) + (NOOBJ_WEIGHT * bce_noobjloss)

        
    def bounding_loss(self, ground_truths, relevant_preds, scale_dim):
        # Broadcast sig and exp across each layer
        relevant_preds[:, [0, 1], :] = torch.sigmoid(relevant_preds[:, [0, 1], :])
        relevant_preds[:, [2, 3], :] = torch.exp(relevant_preds[:, [2, 3], :])

        # Convert to [cell, anchor, features]
        relevant_preds = relevant_preds.permute(0, 2, 1)
        return self.mseloss(ground_truths.repeat(self.num_anchors, 1, 1)[:, :, :4],
                            relevant_preds[:, :, :4])
        
    def CELoss(self, pred, target):
        # pred = (1, 14) that is in raw form
        # target = (1) that is the class id
        return self.celoss(pred, target)
    
    def class_loss(self, target, pred, scale):
        # pred = (5, 21, 13)
        # target = (13, 25)
        # pred_coord = (13, 2)
        target_class = target[:, 5:19] # (13, 14)
        target_class = target_class.repeat(self.num_anchors, 1, 1)
        pred_permute = pred.permute(0, 2, 1) # (5, 13, 21)
        predictions = pred_permute[:, :, 5:19] # (5, 13, 14)
        
        return self.CELoss(predictions, target_class)

    def forward(self, anchors, true_y, pred_y):
        '''
        Loss function for bounding boxes. 
        Assumes predict_boxes in shape [b, x, y, c]. 
        Assumes ground_truths to be original resolution. Loss function will down scale.
        '''
        ground_truths = true_y.permute(2, 0, 1, 3) # Permute to cluster into scales for zipping
        
        total_loss = 0
        for scale_dim, gts, pred_img_scales in zip([13, 26, 52], ground_truths, pred_y):   # iterate through each scale
            bbox_losses = []
            obj_losses = []
            class_losses = []
            for pred_image, image_gt in zip(pred_img_scales, gts):        
                pred_image = torch.stack(torch.split(pred_image, 19, dim=0))
        
                # Get relevant cell coordinates
                gt_valid = image_gt[image_gt[:, 4] != 0] # Use only the ones with objectness = 1 (0 is from padding)
                cell_coords = ((gt_valid[:, [-4, -3]] * (scale_dim - 0.001))).int() # gt % x y * 13 (etc) = pixel position; int = cell position Note: -0.001 to prevent possible chance of it = scale_dim, which is out of range (NEEDS TESTING)
                relevant_preds = pred_image[:, :, cell_coords[:, 0], cell_coords[:, 1]]
                
                # Perform Objectness Loss
                obj_loss = self.ObjLoss(cell_coords.to('cpu'), pred_image.to('cpu'), scale_dim)
                total_loss += obj_loss
                # obj_losses.append(self.ObjLoss(cell_coords, pred_image.to('cpu'), scale_dim))
                # return
                
                # Perform Box Loss
                box_loss = self.bounding_loss(gt_valid.to('cpu'), relevant_preds.to('cpu'), scale_dim)
                total_loss += box_loss
                # bbox_losses.append(self.bounding_loss(gt_valid.to('cpu'), relevant_preds.to('cpu'), scale_dim))
                
                # Perform Class Loss
                class_loss = self.class_loss(gt_valid.to('cpu'), relevant_preds.to('cpu'), scale_dim)
                total_loss += class_loss
                # class_losses.append(self.class_loss(gt_valid.to('cpu'), relevant_preds.to('cpu'), scale_dim))
                
                # Check Negative Loss
                if obj_loss.item() < 0 or box_loss.item() < 0 or class_loss.item() < 0:
                    print(f'Negative Loss Detected. Obj: {obj_loss.item()}, Box: {box_loss.item()}, Class: {class_loss.item()}')
        return total_loss