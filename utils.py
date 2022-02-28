from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

import numpy as np
import torch

import os
import cv2
from glob import glob


class ImageDataSet(Dataset):

    def __init__(self, data_dir, image_channel, transform= None, 
                is_train= True):
        
        super(ImageDataSet, self).__init__()
        self.images_path = np.sort(glob(os.path.join(data_dir, 'images/*.jpeg')))
        self.masks_path = np.sort(glob(os.path.join(data_dir, 'masks/*.jpeg'))) if is_train\
            else None
        
        self.image_channel = image_channel
        self.transform = transform
        self.is_train = is_train
        self.convert_tensor = A.Compose([
                                        ToTensor()
                                        ])
    
    def __getitem__(self, idx):
        if self.is_train:
            img = cv2.cvtColor(cv2.imread(self.images_path[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if self.image_channel == 3 else \
                cv2.imread(self.images_path[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(self.masks_path[idx], cv2.IMREAD_GRAYSCALE)

            h, w = img.shape[:2]
            if h != 224 or w != 224:
                img = cv2.resize(img, (224,224), interpolation= cv2.INTER_CUBIC)
                mask = cv2.resize(mask, (224,224), interpolation= cv2.INTER_CUBIC)
            
            if self.transform is not None:
                transformed = self.transform(image= img, mask= mask)
                return transformed['image'], transformed['mask'] 
            else:
                transformed = self.convert_tensor(image= img, mask= mask)  
                return transformed['image'], transformed['mask']     
        else:
            img = cv2.cvtColor(cv2.imread(self.images_path[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if self.image_channel == 3 else \
                cv2.imread(self.images_path[idx], cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            if h != 224 or w != 224:
                img = cv2.resize(img, (224,224), interpolation= cv2.INTER_AREA)
            transformed = self.convert_tensor(image= img)
            return transformed['image'], False

    def __len__(self):
        return len(self.images_path)


class EarlyStopping(object):

    def __init__(self, 
                patience: int= 10, 
                verbose: bool= False, delta: float= 0, path: str= './checkpoint.pt',
                best_top_k= 4):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta # significant change
        self.path = path

        self.best_score = None
        self.early_stop= False
        self.val_loss_min = np.Inf
        self.counter = 0
        self.best_top_k= best_top_k

    def __call__(self, val_loss, model):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter}/{self.patience}')
            if self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased: {self.val_loss_min:.4f} --> {val_loss:.4f}. Saving model...')
        # torch.save(model.state_dict(), self.path)
        torch.save(model, self.path)
        self.val_loss_min = val_loss


# evaluation measure
def mask_intersection_over_union(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = intersection.sum()/ union.sum()
    return iou_score

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    EPS = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

# # debugging

# transform = A.Compose(
#     [
#       A.HorizontalFlip(p= 0.5),
#       A.RandomRotate90(),
#       A.RandomBrightnessContrast(p=0.5),
#       ToTensorV2()
#     ]
# )

# ds = ImageDataSet('./data/train', 1, transform, is_train= False)
# from torch.utils.data import DataLoader
# dl = DataLoader(ds, 32, True)
# img, mask = next(iter(dl))
# print(img.shape)
# if mask is not None:
#     print(mask.shape)