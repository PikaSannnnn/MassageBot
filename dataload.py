import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from iou import iou
import cv2
import os
import glob

class backDataset(Dataset):
    def __init__(self, transform=None, anchor_boxes = None):
        self.img_paths = glob.glob('dataset/*.png')
        self.transform = transform
        self.anchor_boxes = anchor_boxes * (416/1920)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        original_h, original_w = img.shape[:2]
        normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])
        with open(os.path.join('dataset', 'annotation_percentage', os.path.basename(self.img_paths[idx]).replace('.png', '.txt'))) as f:
            all_boxes = []
            for line in f.readlines():
                class_id, px, py, pw, ph = map(float, line.strip().split(' ')) # class_id, % x, % y, % w, % h
                
                boxes = []
                for scale in [13, 26, 52]:
                    new_box = torch.zeros(25) # create tensor with 4 + 1 + 14 length + 2 (for px, py)

                    # Assign basic values that no need to compute (..., obj, ..., class, ...)
                    new_box[4] = 1  # Automatically at 1.0 because obv. object
                    new_box[int(class_id + 5)] = 1
                    new_box[-2] = pw
                    new_box[-1] = ph

                    # Assign center offset percentage
                    new_box[-4] = px + (pw / 2) # Set x position percentage to gt_bb center
                    new_box[-3] = py + (ph / 2) # Set y position percentage to gt_bb center
                    
                    # Assign raw tw, th offset
                    new_box[[2, 3]] = torch.tensor([pw * scale, ph * scale])
                    
                    # Compute tx, ty based on raw coordinate offset
                    raw_coord = torch.tensor([new_box[-4] * scale, new_box[-3] * scale])
                    cell_coord = raw_coord // 1
                    new_box[[0, 1]] = raw_coord - cell_coord
                    
                    # Get best anchor for this scale                    
                    best_anchor = [None, -1]  # [[anchor_w, anchor_h]_scaled, anchor_iou]
                    for anchor in self.anchor_boxes:
                        anchor_iou = iou([[0, 0, new_box[2], new_box[3]]], [[0, 0] + (anchor / scale).tolist()])
                        if anchor_iou[0] > best_anchor[1]:
                            best_anchor = [anchor / scale, anchor_iou[0]]
                    new_box[[-6, -5]] = best_anchor[0]
                    
                    # Adjust raw tw, th to real tw th
#                     print(new_box[[2, 3]], '|', best_anchor[0], '|', new_box[[2, 3]] / best_anchor[0])
                    new_box[[2, 3]] = new_box[[2, 3]] / best_anchor[0] # This will give the exp(tw) and exp(th) to get from anchor to gt

                    boxes.append(new_box)
                all_boxes.append(torch.stack(boxes))
#                 print(all_boxes[-1])
        img = normalize(img)
        return img, torch.stack(all_boxes)

def collate_fn(batch): # each item in batch is (x, y)
    img = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    max_boxes = max([len(box) for box in boxes])
    boxes_list_padded = []
    for box in boxes:
        if box.shape[0] < max_boxes:
            box = torch.cat([box, torch.zeros(max_boxes - box.shape[0], box.shape[1], box.shape[2])])
        boxes_list_padded.append(box)
    return torch.stack(img), torch.stack(boxes_list_padded)