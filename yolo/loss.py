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
    def __init__(self, num_anchors, batch_avg=False):
        super().__init__()
        self.num_anchors = num_anchors
        self.batch_avg = batch_avg        
        
        self.mseloss = nn.MSELoss()

    def ObjDiceLoss(self, target_mask, pred_mask):
        smooth = 1e-6
        intersection = (target_mask * pred_mask).sum() + smooth
        union = target_mask.sum() + pred_mask.sum() + smooth
        Dice_score = 2 * intersection / union
        loss = 1 - Dice_score
        return loss

    def ObjLoss(self, target_coords, preds, scale):
        assert preds.shape[0] == self.num_anchors, f'prediction should have {self.num_anchors} anchors'
        
        loss = 0
        target_OH = torch.zeros(scale, scale)
        target_OH[target_coords[:,1], target_coords[:,0]] = 1
        target_OH = target_OH.view(-1)
        target_OH = target_OH.repeat(self.num_anchors, 1)
        
        pred_OH = preds[:, 4, :, :]
        pred_OH = pred_OH.view(self.num_anchors, -1)
        pred_OH = torch.sigmoid(pred_OH)
        # print(pred_OH.shape)
        # print(pred_OH)

        # target_OH = torch.zeros(scale, scale)
        # target_OH = one_hot_coordinate(target, scale).view(-1)
        return F.binary_cross_entropy(target_OH, pred_OH)
        # return self.ObjDiceLoss(target_OH, pred_OH)
        
    def bounding_loss(self, ground_truths, relevant_preds, scale_dim):
        # Broadcast sig and exp across each layer
        relevant_preds[:, [0, 1], :] = torch.sigmoid(relevant_preds[:, [0, 1], :])
        relevant_preds[:, [2, 3], :] = torch.exp(relevant_preds[:, [2, 3], :])

        # Convert to [cell, anchor, features]
        relevant_preds = relevant_preds.permute(0, 2, 1)
#         relevant_preds = relevant_preds.permute(2, 0, 1)
#         print(relevant_preds.shape)
#         print(relevant_preds[:, :, :4].shape)

        # Compute MSE Losses
#         print(ground_truths.shape)
#         print(ground_truths[:, :4])
#         print(torch.flatten(ground_truths[:, :4]).shape)
#         test = ground_truths.repeat(self.num_anchors, 1, 1)
#         print(test[:, :, :4].shape, relevant_preds[:, :, :4].shape)
#         loss_compare = [test[:, :, :4], relevant_preds[:, :, :4]]

        
        # DEBUG: Keeping to compare performance
#         mse_losses = []
#         for pred in relevant_preds:
#             for box in pred:
# #                 print(box.shape)
#                 mse_losses.append(torch.sum(torch.tensor([self.mseloss(box[0:4], gt[0:4]) for gt in ground_truths])))
        
#         return torch.mean(torch.tensor(mse_losses))
        return self.mseloss(ground_truths.repeat(self.num_anchors, 1, 1)[:, :, :4],
                            relevant_preds[:, :, :4])

    def forward(self, anchors, true_y, pred_y):
        '''
        Loss function for bounding boxes. 
        Assumes predict_boxes in shape [b, x, y, c]. 
        Assumes ground_truths to be original resolution. Loss function will down scale.
        '''
#         print(true_y.shape)
        ground_truths = true_y.permute(2, 0, 1, 3) # Permute to cluster into scales for zipping
#         print(ground_truths.shape)
        
        total_loss = 0
        for scale_dim, gts, pred_img_scales in zip([13, 26, 52], ground_truths, pred_y):   # iterate through each scale
#             print(f'Scale: 416x416 -> {scale_dim}x{scale_dim}')
#             print(img_scales.shape)
#             print(gts.shape)
            bbox_losses = []
            obj_losses = []
            for pred_image, image_gt in zip(pred_img_scales, gts):
                # anchors // (416 // scale_dim)
        #         img_scale = img_scale.permute(1, 2, 0)
        #         img_scale = img_scale.view(95, -1)  # DEBUG: temp flatten for testing
        
                pred_image = torch.stack(torch.split(pred_image, 19, dim=0))
        

                # Get relevant cell coordinates
                gt_valid = image_gt[image_gt[:, 4] != 0] # Use only the ones with objectness = 1 (0 is from padding)
                cell_coords = ((gt_valid[:, [-4, -3]] * scale_dim) - 1).int() # gt % x y * 13 (etc) = pixel position; int = cell position
        #         cell_coords = cell_coords[:, 0] * scale_dim + cell_coords[:, 1] # DEBUG: temp convert to flattened idx
                
                # Perform Objectness Loss
                # self.obj_loss(image_gt.to('cpu'), pred_image.to('cpu'), scale_dim)
                obj_losses.append(self.ObjLoss(cell_coords, pred_image.to('cpu'), scale_dim))
                # print(obj_losses[-1])
                # return
        
            
                # print('c', cell_coords.shape)
                # print('t', test[:, :, cell_coords[:, 0], cell_coords[:, 1]].shape)
                relevant_preds = pred_image[:, :, cell_coords[:, 0], cell_coords[:, 1]]
                # print('rb', relevant_preds.shape)
                # for i, (coord, gt) in enumerate(zip(cell_coords, gt_valid)):
                #     print(coord, gt)
                #     print(relevant_preds[:, :, i])
                #     print('-'*30)
                #     break
                # print(relevant_preds[0, :, 1])
                # print(cell_coords[0])
                # print(pred_image[:, :, cell_coords[0, 0], cell_coords[0, 1]])
                
                # relevant_preds = torch.stack(torch.split(relevant_preds, 19, dim=0)) # DEBUG: Need to remove, no longer needed theoretically
                
                # NOTE: PRINT THIS TO SEE WHATS GOING ON
                # print(gt_valid.shape)
                # print(gt_valid)
                # print('HERE BE THE OTHER')
                # print(relevant_preds.shape)
                # print(relevant_preds)
                
                # Perform Box Loss
                bbox_losses.append(self.bounding_loss(gt_valid.to('cpu'), relevant_preds.to('cpu'), scale_dim))
                
            # Mean across batches
            bbox_losses = torch.tensor(bbox_losses)
            bbox_losses = torch.mean(bbox_losses) if self.batch_avg else torch.sum(bbox_losses)
            
            obj_losses = torch.tensor(obj_losses)
            obj_losses = torch.mean(obj_losses) if self.batch_avg else torch.sum(obj_losses)
            
            # Add the total los for this scale
            total_loss += bbox_losses + obj_losses
        return total_loss
    
    def backward(sef):
        return