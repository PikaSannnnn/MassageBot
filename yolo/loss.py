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
        
    def function_to_grid(ground_truths,S):
        output_size = [13, 26, 52]
        S_dim = output_size[S]
        return torch.round(ground_truths[:,:,S,-4:-2] * S_dim).long()
    
    def one_hot_coordinate(self, ground_truths,S):
        output_size = [13, 26, 52]
        S_dim = output_size[S]
        grid_have_stuff = function_to_grid(ground_truths,S)
        one_hot = torch.zeros(10,S_dim,S_dim)
        for i in range(10):
            one_hot[i,grid_have_stuff[i,:,1],grid_have_stuff[i,:,0]] = 1
        return one_hot

    def obj_loss(self, target, pred, scale_dim):
        # torch.Size([10, 95, 13, 13]) torch.Size([10, 14, 3, 25])
        # torch.Size([5, 19, 13, 13]) torch.Size([14, 25])
        print(pred.shape, target.shape)
        # test = target.repeat(5, 1)
        # print(torch.flatten(test))
        
        # pred = pred.reshape(13, 13, -1, 19)
        print(torch.flatten(pred).shape)
        
        pred0 = pred[0].reshape(10, 13, 13, -1, 19)
        pred1 = pred[1].reshape(10, 26, 26, -1, 19)
        pred2 = pred[2].reshape(10, 52, 52, -1, 19)
        pred0 = pred0[:,:,:,:,4].reshape(-1)
        pred1 = pred1[:,:,:,:,4].reshape(-1)
        pred2 = pred2[:,:,:,:,4].reshape(-1)
        
        print(pred0.shape, target.shape)
        return

        OH_gt0 = one_hot_coordinate(target,0).reshape(-1).repeat(5)
        OH_gt1 = one_hot_coordinate(target,1).reshape(-1).repeat(5)
        OH_gt2 = one_hot_coordinate(target,2).reshape(-1).repeat(5)
        
        # Apply sigmoid activation to the raw predictions
        pred0 = torch.sigmoid(pred0)
        pred1 = torch.sigmoid(pred1)
        pred2 = torch.sigmoid(pred2)
        
        # Compute the binary cross entropy loss
        loss = F.binary_cross_entropy(pred0, OH_gt0)
        loss += F.binary_cross_entropy(pred1, OH_gt1)
        loss += F.binary_cross_entropy(pred2, OH_gt2)
        
        return loss
        
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
            obj_losses = [0.0]
            for pred_image, image_gt in zip(pred_img_scales, gts):
                # anchors // (416 // scale_dim)
        #         img_scale = img_scale.permute(1, 2, 0)
        #         img_scale = img_scale.view(95, -1)  # DEBUG: temp flatten for testing
        
                pred_image = torch.stack(torch.split(pred_image, 19, dim=0))
        
                # Perform Objectness Loss
                self.obj_loss(image_gt.to('cpu'), pred_image.to('cpu'), scale_dim)

                # Get relevant cell coordinates
                gt_valid = image_gt[image_gt[:, 4] != 0] # Use only the ones with objectness = 1 (0 is from padding)
                cell_coords = ((gt_valid[:, [-4, -3]] * scale_dim) - 1).int() # gt % x y * 13 (etc) = pixel position; int = cell position
        #         cell_coords = cell_coords[:, 0] * scale_dim + cell_coords[:, 1] # DEBUG: temp convert to flattened idx

                # print('c', cell_coords.shape)
                # print('t', test[:, :, cell_coords[:, 0], cell_coords[:, 1]].shape)
                relevant_preds = pred_image[:, :, cell_coords[:, 0], cell_coords[:, 1]]
                # print('rb', relevant_preds.shape)
                
                # relevant_preds = torch.stack(torch.split(relevant_preds, 19, dim=0)) # DEBUG: Need to remove, no longer needed theoretically
                
                # NOTE: PRINT THIS TO SEE WHATS GOING ON
                print(gt_valid.shape)
                # print(gt_valid)
                # print('HERE BE THE OTHER')
                # print(relevant_preds.shape)
                # print(relevant_preds)
                
                # Perform Box Loss
                bbox_losses.append(self.bounding_loss(gt_valid.to('cpu'), relevant_preds.to('cpu'), scale_dim))
                
                # obj_losses.append()
                
                return
                
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