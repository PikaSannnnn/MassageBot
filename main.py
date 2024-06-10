# Main modele imports
import os
import json
import cv2
import numpy as np
import torch
import glob
import torch.nn as nn

# Util imports
import tqdm # For progress bars
import matplotlib.patches as patches # For plotting boxes of results
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import sys
from datetime import datetime
import pickle

# Yolo imports
from yolo.yolo import Yolo
from yolo.loss import MultiFactorLoss
import yolo.training as yolo_tv
from dataload import *

###############
#### CODE START
###############
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using: ', device)

class_dict = {
    'Head': 0,
    'RShoulder': 1,
    'LShoulder': 2,
    'LElbow': 3,
    'LWrist': 4,
    'RHip': 5,
    'LHip': 6,
    'LKnee': 7,
    'RKnee': 8,
    'RAnkle': 9,
    'LAnkle': 10,
    'RElbow': 11,
    'RWrist': 12,
    'Neck': 13
}
num_classes = len(class_dict.keys())

anchors = torch.tensor([[151.37092679, 151.37092679], [116.1775567, 116.1775567 ], [ 69.10933737,  69.10933737], 
                        [206.27557981, 206.27557981], [257.97346109, 257.97346109]])
num_anchors = anchors.shape[0]

batch_epochs = [(10, 20), (32, 20)]
loss_params = [(1, 10),(0.1, 20), (0.5, 25), (0.3, 20)]
lrs = [1e-4, 5e-4]
# batch_epochs = [(10, 20)]
# loss_params = [(0.3, 20)]
# lrs = [1e-4]

model_dir = os.path.join(os.getcwd(), 'models')
out_img_dir = os.path.join(os.getcwd(), 'model_imgs')

###############
#### UTIL STUFF
###############
normal_size = (128,128)

def load_image(path):
    return cv2.imread(path)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def crop_image(image, coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    return image[int(y1):int(y2), int(x1):int(x2)]

def show_image_detail(image, json):
    plt.imshow(image)
    fig, axs = plt.subplots(1, len(json['shapes']), figsize=(15, 15))
    for i, shape in enumerate(json['shapes']):
        croped_test_img = crop_image(image, shape['points'])
        croped_test_img = resize_image(croped_test_img)
        print(croped_test_img.shape)
        axs[i].imshow(croped_test_img)
        axs[i].set_title(shape['label'])
    plt.show()

def resize_image(image, size=normal_size):
    return cv2.resize(image, size)

def vaildate_yolo_formate(path):
    imag = cv2.imread(path)
    yolo_path = os.path.join('dataset', 'annotation_percentage', os.path.basename(path).split('.')[0] + '.txt')
    with open(yolo_path, 'r') as boxes:
        for box in boxes:
            box = box.split(' ')
            class_id, x_center, y_center, width, height = map(float, box)
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            cv2.rectangle(imag, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imag, str(int(class_id)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
    imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
    plt.imshow(imag)
    plt.show()

def show_img_box(img, data, img_path=None):
    tmp_img = img.permute(1, 2, 0)
    tmp_img = cv2.cvtColor(tmp_img.numpy(), cv2.COLOR_BGR2RGB)
    for box in data:
        class_id = torch.argmax(box[0][5:19])
        box_dim = (box[0][[-2, -1]] * 416)
        x1, y1 = (box[0][[-4, -3]] * 416) - (box_dim / 2) 
        x2, y2 = (box[0][[-4, -3]] * 416) + (box_dim / 2) 
        cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(tmp_img, str(int(class_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    plt.imshow(tmp_img)
    
    if img_path != None: 
        plt.savefig(img_path)
    
def plot_pic_with_box(img, pred, Scale, imgIdx,anchorIdx=0, img_size=416, threshold=0.5, img_path=None):
    anchors = torch.tensor([[151.37092679, 151.37092679], [116.1775567, 116.1775567 ], [ 69.10933737,  69.10933737], 
                        [206.27557981, 206.27557981], [257.97346109, 257.97346109]]).cpu()
    anchors_size = anchors[anchorIdx]
    grid_size = torch.tensor([Scale, Scale]).cpu()
    anchors_dim = torch.tensor([1920, 1920]).cpu()
    anchors_yolo = anchors_size * (grid_size / anchors_dim).cpu()
    
    # Prediction Processing
    pred_reshape = pred.reshape(10, 5, 19, Scale, Scale).cpu()
    pred_reshape = pred_reshape.permute(0, 3, 4, 1, 2).cpu()
    output = pred_reshape[:,:,:,anchorIdx,:].cpu() # (10, 13, 13, 19)
    box_xy = output[..., :2].detach().cpu() # (10, 13, 13, 2)
    box_wh = output[..., 2:4].cpu() # (10, 13, 13, 2)
    
    scale_ratio = Scale / img_size # Scaler
    box_wh = torch.exp(box_wh).cpu() * anchors_yolo.cpu()
    box_wh = box_wh.detach().cpu()
    box_xy = torch.sigmoid(box_xy).detach().numpy()

    # Draw Image with Boxes
    fig,ax = plt.subplots(1)
    ax.imshow(img.cpu().permute(1, 2, 0))
    for i in range(Scale):
        for j in range(Scale):
            if torch.sigmoid(output[imgIdx, i, j, 4]) > threshold:
                box_truexy = np.array([i, j]) # Get cell in normal image
                box_truexy = (box_truexy + box_xy[imgIdx, i, j, :]) / scale_ratio # Add offset and scale back up to 416      
                box_truewh = np.array([box_wh[imgIdx, i, j, 0], box_wh[imgIdx, i, j, 1]]) / scale_ratio # Get and scale wh back up to 416
                box_truexy -= (box_truewh / 2) # Shift to top left for box drawing
                box = patches.Rectangle((box_truexy[0], box_truexy[1]), box_truewh[0], box_truewh[1], linewidth=1, edgecolor='r', facecolor='none')

                ax.add_patch(box)
                
    if img_path != None: 
        fig.savefig(img_path)

###############
#### MAIN STUFF
###############
def setup():
    if not os.path.exists(model_dir):
        print('Creating', os.path.basename(model_dir), 'directory in', os.path.dirname(model_dir))
        os.makedirs(model_dir)
    else:
        print(model_dir, 'exists :D')
        
    if not os.path.exists(out_img_dir):
        print('Creating', os.path.basename(out_img_dir), 'directory in', os.path.dirname(out_img_dir))
        os.makedirs(out_img_dir)
    else:
        print(out_img_dir, 'exists :D')

def get_loaders(train_split=0.7, val_split=0.1):
    dataset = backDataset(anchor_boxes=anchors)
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader
    
def train(overwrite=False):
    for num_batches, num_epochs in batch_epochs:
        for loss_param in loss_params:
            for lr in lrs:
                model_file_base = f'PosteriorYolo_b{num_batches}e{num_epochs}l{loss_param[0]}-{loss_param[1]}r{lr}'
                model_file = os.path.join(model_dir, f'{model_file_base}.pth')
                    
                # with open(os.path.join(model_dir, f'{model_file_base}_train_loss.dat'), 'rb') as file:
                #     train_losses = pickle.load(file)
                #     print(train_losses)
                
                if os.path.exists(model_file) and not overwrite:
                    print(os.path.basename(model_dir), 'already exists, skipping. If you\'d like to overwrite it, set -w flag.')
                    continue
                
                train_loader, val_loader, test_loader = get_loaders()
                
                model = Yolo(num_classes, num_anchors, debug=False)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
                criterion_fn = MultiFactorLoss(5, obj_weights=loss_param)
                criterion_fn.requires_grad = True
                for param in model.parameters():
                    assert param.requires_grad
                    
                try:
                    train_loss, val_loss = yolo_tv.train(model, optimizer, criterion_fn, train_loader, val_loader, anchors, epochs=num_epochs)
                    with open(os.path.join(model_dir, f'{model_file_base}_train_loss.dat'), 'wb') as file:
                        pickle.dump(train_loss, file)
                        
                    with open(os.path.join(model_dir, f'{model_file_base}_val_loss.dat'), 'wb') as file:
                        pickle.dump(val_loss, file)
                    # train_losses.append(t_loss)
                    # val_losses.append(v_loss)
                except Exception as e:
                    print(e)
                else:
                    print('Saving model to', model_file)
                    torch.save(model, model_file)
                finally: # Delete to clear up gpu for next train/runs
                    del model
                    del criterion_fn
                    del optimizer
                    
def test(model_file, num_imgs, thresholds=(0.3, 0.3, 0.3), save_img=False):
    assert os.path.exists(os.path.join(model_dir, model_file)), 'model does not exist'
    assert len(thresholds) == 3, 'thresholds expected to be of size 3'
    
    current_datetime = datetime.now()
    date_file_prefix = model_file + current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    
    if save_img:
        img_path = os.path.join(out_img_dir, date_file_prefix)
        os.makedirs(img_path)
    else:
        img_path = None
    
    model = torch.load(os.path.join(model_dir, model_file))
    model.to(device)
#     model.to('cpu')
    
    _, _, test_loader = get_loaders()
    
    img, data = next(iter(test_loader))
    test_result = model(img.to(device))
#     test_result = model(img.to('cpu'))
    for i in range(num_imgs):
        gt_img_path = None if img_path == None else os.path.join(img_path, f'{i}_gt.png')            
        show_img_box(img[i % 10], data[i % 10], img_path=gt_img_path) # Plot expected image
        
        for j, (scale, threshold) in enumerate(zip([13, 26, 52], thresholds)):
            pred_img_path = None if img_path == None else os.path.join(img_path, f'{i}_pred_{scale}.png')
            plot_pic_with_box(img[i % 10], test_result[j], scale, i % 10, threshold=threshold, img_path=pred_img_path)
        
        
        if (i + 1) % 10 == 0: # Retrain for next 10 imgs
            img, data = next(iter(test_loader))
            test_result = model(img.to(device))
#             test_result = model(img.to('cpu'))

def main():
    setup()
    valid_input = False
    
    if '-t' in sys.argv:
        overwrite = True if '-w' in sys.argv else False
        print('Train Overwrite:', overwrite)
        train(overwrite)
        valid_input = True

    if '-p' in sys.argv:
        model_list = glob.glob(os.path.join(model_dir, '*.pth'))
        
        if len(model_list) == 0:
            print('No Model Files')
            return
        
        while True:
            print('')
            print('Model Files:')
            [print(f'\t{i}.', os.path.basename(model_filename)) for i, model_filename in enumerate(model_list)]
            
            model_idx = int(input('Which model would you like to load (#):'))
            if model_idx < len(model_list):
                break
            else:
                print('Please choose a valid number!')
            
        num_imgs = int(input('How many images:'))
        # thresholds = input('Give 3 thresholds (Leave empty for default)').split()
        save_img = False if 'n' in input('Save imgs (y/n)? (Defaults yes):') else True
        
        test(os.path.basename(model_list[model_idx]), num_imgs, save_img=save_img)
        valid_input = True
        
    if not valid_input:
        print('Invalid usage. Expected `python main.py <flags>`. Flags:\n\t-t: train\n\t-w: overwrite train model files\n\t-p: test with imgs')
        

if __name__ == "__main__":
    main()