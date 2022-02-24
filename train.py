import os
import sys
import pickle

import segmentation_models_pytorch as smp
import argparse

import torch 
from torch.utils.data import DataLoader
from torch import nn
from torchsummary import summary
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

from utils import *

def argparser():
    p = argparse.ArgumentParser()
    
    # data args
    p.add_argument('--train_path', type= str, required= True,
                help= 'a path to training data')
    p.add_argument('--valid_path', type= str, required= False, 
                help= 'a path to validation data')
   
    # data augmentation options
    p.add_argument('--rot_prob', type= float, default= 0.2,
                help= 'probability of 90 degree rotation degree, default= 0.2')
    p.add_argument('--h_flip_prob', type= float, default= 0.2,
                help= 'probability of horizontal flip, default= 0.2')
    p.add_argument('--bright_contrast_prob', type= float, default= 0.1,
                help= 'probability of RandomBrightnessContrast')
    p.add_argument('--b_limit', type= float, default= 0.2, 
                help= 'brightness_limit of RandomBrightnessContrast, default= 0.2')
    p.add_argument('--c_limit', type= float, default= 0.2, 
                help= 'contrast_limit of RandomBrightnessContrast, default= 0.2')
    p.add_argument('--elatic_prob', type= float, default= 0.2,
                help= 'probability of ElasticTransform, default= 0.2')

    # model configs
    p.add_argument('--model_save_path', type= str, default= './model',
                help= 'a path to save a model (folder)')
    p.add_argument('--model_path', type= str, required= False,
                help= 'a path to a trained model if any (.pth)')
    p.add_argument('--model_type', type= int, default= 0,
                help='0:Unet, 1:Unet++, 2:DeepLabV3+')
    p.add_argument('--encoder_name', type= str, default= 'resnet34',
                help= 'choose encoder, (ex) resnet18, resnet34, resnet50 ... resnet152, xception, inceptionv4 ....\n\
                    refer to https://github.com/qubvel/segmentation_models.pytorch#encoders for the details')
    p.add_argument('--encoder_weights', type= str, default= 'imagenet', 
                help= 'imagenet/ssel/swsl ...')
    p.add_argument('--in_channels', type= int, required= True, 
                help= 'model input channels (1 for gray-scale images, 3 for RGB, etc.)')
    p.add_argument('--classes', type= int, required= True, 
                help= 'model output channels (number of classes in your dataset)')

    
    # training configs
    p.add_argument('--max_epoches', type= int, required= True,
                help= 'max number of epoches')
    p.add_argument('--batch_size', type= int, required= True,
                help= 'batch size')
    p.add_argument('--lr', type= float, default= 1e-3, 
                help= 'learning rate')
    p.add_argument('--print_log_option', type= int, default= 5,
                help= 'print batch loss every \'print_log_option\' step')
    ## early stopping conditions
    p.add_argument('--patience', type= int, default= 10, 
                help= 'patience of early stopping')
    p.add_argument('--delta', type= float, default= 0,
                help= 'delta: significant change of early stopping condition')
    p.add_argument('--best_top_k', type= int, default= 4,
                help= 'the number of models to save during training')
    config = p.parse_args()
    return config

def main(config):

    # make directories
    os.makedirs(config.model_save_path, exist_ok= True)

    # device setting
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # transform option
    if config.in_channels == 3:
        transform = A.Compose(
                                [
                                A.HorizontalFlip(p=config.h_flip_prob),
                                A.RandomRotate90(p= config.rot_prob),
                                A.RandomBrightnessContrast(
                                    brightness_limit= config.b_limit, contrast_limit= config.c_limit,
                                    p=config.bright_contrast_prob),
                                A.ElasticTransform(config.elatic_prob),
                                ToTensor()
                                ]
                            )
    else:
        transform = A.Compose(
                                [
                                A.HorizontalFlip(p=config.h_flip_prob),
                                A.RandomRotate90(p= config.rot_prob),
                                A.RandomBrightnessContrast(
                                    brightness_limit= config.b_limit, contrast_limit= config.c_limit,
                                    p=config.bright_contrast_prob),
                                A.ElasticTransform(config.elatic_prob),
                                ToTensor()
                                ]
                            )     
    
    # read data
    train_ds = ImageDataSet(config.train_path, config.in_channels, transform)
    valid_ds = ImageDataSet(config.valid_path, config.in_channels)
    train_dl = DataLoader(train_ds, config.batch_size, shuffle= True)
    valid_dl = DataLoader(valid_ds, config.batch_size, shuffle= False)

    # model
    # 0:Unet, 1:Unet++, 2:DeepLabV3+
    # todo: if config.model_path is not None:
    # model.load(config.model_path)
    if config.model_path is None:
        aux_params=dict(
                        pooling='max',             # one of 'avg', 'max'
                        dropout=0.5,               # dropout ratio, default is None
                        # activation='sigmoid',      # activation function, default is None
                        classes=config.classes,                 # define number of output labels
                        )
        if config.model_type == 0:
            model = smp.Unet(
                encoder_name= config.encoder_name,
                encoder_weights= config.encoder_weights,
                in_channels= config.in_channels,
                classes= config.classes,
                aux_params= aux_params
            ).to(device)  
        elif config.model_type == 1:
            model = smp.UnetPlusPlus(
                encoder_name= config.encoder_name,
                encoder_weights= config.encoder_weights,
                in_channels= config.in_channels,
                classes= config.classes,
                aux_params= aux_params
            ).to(device)  
        elif config.model_type == 2:
            model = smp.DeepLabV3Plus(
                encoder_name= config.encoder_name,
                encoder_weights= config.encoder_weights,
                in_channels= config.in_channels,
                classes= config.classes,
                aux_params= aux_params
            ).to(device)       
        else:
            model= None
            print('--(!)This repo does not support the model yet...')
            sys.exit()
    
    else:
        model = torch.load(config.model_path).to(device)

    print(summary(model, (config.in_channels, 512, 512)))
    print(f'device: {device}')
    # training
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    criterion = nn.CrossEntropyLoss() if config.classes > 1 else nn.BCEWithLogitsLoss() # reduction= mean

    early_stopping = EarlyStopping(
        patience= config.patience,
        verbose= True,
        delta = config.delta,
        path= os.path.join(config.model_save_path,f'checkpoint-{config.in_channels}_{config.classes}-model_type_{config.model_type}.pth'),
        best_top_k= config.best_top_k
    )

    logs = {
        'tr_loss':[],
        'valid_loss':[]
    }
    print('Start training...')
    for epoch in range(config.max_epoches):
        # to store losses per epoch
        tr_loss, valid_loss = 0, 0
        # a training loop
        for batch_idx, (img, mask) in enumerate(train_dl):

            img, mask = img.to(device), mask.to(device) 

            model.train()
            # feed forward
            with torch.set_grad_enabled(True):
                predicted_mask, _ = model(img)
                loss = criterion(predicted_mask, mask)
            
            # backward 
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store the d_tr_loss
            tr_loss += loss.detach().cpu().item()

            if (batch_idx+1) % config.print_log_option == 0:
                print(f'Epoch [{epoch+1}/{config.max_epoches}] Batch [{batch_idx+1}/{config.batch_size}]: \
                    loss = {loss.detach().cpu().item()}')

        # a validation loop 
        for batch_idx, (img, mask) in enumerate(valid_dl):

            img, mask = img.to(device), mask.to(device)
            
            model.eval()
            with torch.no_grad():
                predicted_mask, _ = model(img)
                loss = criterion(predicted_mask, mask)
            valid_loss += loss.detach().cpu().item()
        
        # save current loss values
        tr_loss, valid_loss = tr_loss/config.batch_size, valid_loss/config.batch_size
        logs['tr_loss'].append(tr_loss)
        logs['valid_loss'].append(valid_loss)

        print(f'Epoch [{epoch+1}/{config.max_epoches}]: training loss= {tr_loss:.6f}, validation loss= {valid_loss:.4f}')
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            break 

    torch.save(model, os.path.join(config.model_save_path,f'final_model-{config.in_channels}_{config.classes}-model_type_{config.model_type}.pth') )

    log_path= os.path.join(config.model_save_path, 'training_logs')
    os.makedirs(log_path, exist_ok= True)
    log_file_path= os.path.join(log_path, 'training_logs.pickle')
    with open(log_file_path, 'wb') as f: 
        pickle.dump(logs, f)
    

if __name__ == '__main__':
    config = argparser()
    main(config)
    
    