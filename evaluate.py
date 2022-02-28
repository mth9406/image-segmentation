import os
import cv2
from torchsummary import summary

import argparse

import torch 
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn

from utils import *

def argparser():
    p = argparse.ArgumentParser()
    
    # data args
    p.add_argument('--test_path', type= str, required= True,
                help= 'a path to training data')
    p.add_argument('--in_channels', type= int, required= True,
                help= 'channel size of an image')
    p.add_argument('--classes', type= int, required= True,
                help= 'the number of classes to predict')

    # model configs
    p.add_argument('--model_dir', type= str, required= True, 
                help= 'a directory to the model (model path)')
    p.add_argument('--model_path', type= str, required= True,
                help= 'a path to a trained model if any (.pth, ...)')
    p.add_argument('--batch_size', type= int, default= 32,
                help= 'batch size')
    
    # a path to save results
    config = p.parse_args()
    return config


def main(config):

    predictions_path = os.path.join(config.model_dir, 'predictions')
    rois_path = os.path.join(config.model_dir, 'rois')
    os.makedirs(predictions_path, exist_ok= True)
    os.makedirs(rois_path, exist_ok= True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = torch.load(config.model_path).to(device)
    print(summary(model, (config.in_channels, 224, 224)))

    ds = ImageDataSet(config.test_path, config.in_channels)
    dl = DataLoader(ds, batch_size= config.batch_size, shuffle= False)

    criterion = nn.CrossEntropyLoss() if config.classes > 1 else nn.BCEWithLogitsLoss()

    print('making predictions ...')
    # a validation loop 
    valid_loss = {
        'loss':[],
        'miou':[]
    }

    temp = os.path.join(config.test_path, 'masks')
    temp2 = os.path.join(config.test_path, 'images')
    file_names = list(map(lambda x:x.replace(temp, predictions_path), ds.masks_path))
    roi_names = list(map(lambda x:x.replace(temp2, rois_path), ds.images_path))

    print(f'Example path to a prediction file: {file_names[0]}')
    print(f'Example path to a roi file: {roi_names[0]}')

    predictions = []
    rois = []
    to_prob = F.sigmoid if config.classes == 1 else F.softmax
    num_batches = int(np.ceil(len(file_names)/config.batch_size))
    for batch_idx, (img, mask) in enumerate(dl):

        img, mask = img.to(device), mask.to(device) # bs, 3, h, w // bs, 1, h, w
        bs, c, h, w= img.shape
        roi = torch.zeros((bs, config.in_channels, h, w), dtype= torch.uint8) # bs, 3, h, w
        model.eval()
        with torch.no_grad():
            predicted_mask, label = model(img)
            loss = criterion(predicted_mask, mask)
        p = predicted_mask.detach().cpu()
        p = to_prob(p) # bs, 1, h, w
        if config.classes == 1:
            p[p > 0.5] = 255
            p[p <= 0.5] = 0
            p = p.type(torch.uint8) # torch
            p_indx = p.repeat(1,3,1,1)
            roi[p_indx > 0] = (255.*img.detach().cpu()[p_indx > 0]).type(torch.uint8) # bs, 3, h, w
        else:
            p = torch.argmax(p) 
            # todo
            # need color map...

        print(f'Batch [{batch_idx+1}/{num_batches}]: saving validation loss and miou...')
        valid_loss['loss'].append(loss.detach().cpu().item()) 
        mask_2 = (mask * 255.).detach().cpu().type(torch.uint8) # torch
        miou = iou_pytorch(p, mask_2).detach().cpu().item()
        valid_loss['miou'].append(miou)

        print(f'Batch [{batch_idx+1}/{num_batches}]: converting torch to images and save in {predictions_path}...')

        roi = np.transpose(roi.numpy(), (0,2,3,1)) # bs, h, w, 3; numpy; roi
        p = np.transpose(p.numpy(), (0,2,3,1)).squeeze() # bs, h, w; numpy; prediction 

        for i in range(bs):
            # save roi and predictions
            idx = config.batch_size*batch_idx + i
            cv2.imwrite(file_names[idx], p[i])
            cv2.imwrite(roi_names[idx], roi[i])

        # predictions.append(p)
        # rois.append(roi)
        
    # predictions = torch.cat(predictions, dim= 0)
    # rois = torch.cat(rois, axis= 0)
    valid_loss['loss'] = np.mean(valid_loss['loss'])
    valid_loss['miou'] = np.mean(valid_loss['miou'])

    test_score_path = os.path.join(config.model_dir, 'test_score.txt')

    with open(test_score_path, 'w') as f:   
        for k, v in valid_loss.items():
            f.write(f'{k}:{v}\n')

if __name__ == '__main__':
    config = argparser()
    main(config)
