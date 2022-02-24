import os
from glob import glob
import sys
import cv2
from utils import *
import argparse
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
import numpy as np

# read test data
# and predicts its RoI in a real time.
def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path', type= str, required= True,
                help= 'a path to test data, required= True')
    p.add_argument('--image_format', type= str, default= 'png',
                help= 'image format (ex) jpg, png, jpeg, ...')
    p.add_argument('--model_path', type= str, required= True, 
                help= 'a model file (.pth)')
    p.add_argument('--classes', type= int, required= True,
                help= 'the number of classes to predict')
    config = p.parse_args()
    return config

def imread(img, code= cv2.IMREAD_COLOR):
    src = cv2.imread(img, code)
    if src is None:
        print('Image load failed!')
        sys.exit()
    return src

def main(config):
    
    # images and masks
    images_path = np.sort(glob(os.path.join(config.test_path, 'images/*.'+config.image_format)))
    masks_path = np.sort(glob(os.path.join(config.test_path, 'masks/*.'+config.image_format)))
    convert_tensor = ToTensor()
    # model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = torch.load(config.model_path).to(device)

    # for i in range(len(images_path)):
    for i in range(len(images_path)):
        img, mask = imread(images_path[i]), imread(masks_path[i], cv2.IMREAD_GRAYSCALE)
        roi = np.zeros(img.shape[:2], dtype= np.uint8)
        transformed = convert_tensor(image= img, mask= mask)
        img, mask = transformed['image'].to(device), transformed['mask'].to(device)
        
        predicted_mask, _ = model(img)
        if config.classes == 1:

    

if __name__ == '__main__':
    config = argparser()
    main(config)
