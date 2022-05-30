# image-segmentation

A PyTorch implementation of the u-net + ViT..

Data used in this repo is CVC data.    
CVC-ClinicDB is a database of frames extracted from colonoscopy videos     
The dataset contains several examples of polyp frames & corresponding ground truth for them.    
The Ground Truth images consists of a mask corresponding to the region covered by the polyp in the image.

# How to use
First of all, download the data from the data source above.    
and then, run main.py. Below describes how to train a model using main.py.

```bash
usage: train.py [-h] --train_path TRAIN_PATH [--valid_path VALID_PATH]
                [--rot_prob ROT_PROB] [--h_flip_prob H_FLIP_PROB]
                [--bright_contrast_prob BRIGHT_CONTRAST_PROB]
                [--b_limit B_LIMIT] [--c_limit C_LIMIT]
                [--elatic_prob ELATIC_PROB]
                [--model_save_path MODEL_SAVE_PATH] [--model_path MODEL_PATH]
                [--model_type MODEL_TYPE] [--encoder_name ENCODER_NAME]
                [--encoder_weights ENCODER_WEIGHTS] --in_channels IN_CHANNELS
                --classes CLASSES --max_epoches MAX_EPOCHES --batch_size
                BATCH_SIZE [--lr LR] [--print_log_option PRINT_LOG_OPTION]
                [--patience PATIENCE] [--delta DELTA]
                [--best_top_k BEST_TOP_K]
```

# Optional arguments
```
optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        a path to training data
  --valid_path VALID_PATH
                        a path to validation data
  --rot_prob ROT_PROB   probability of 90 degree rotation degree, default= 0.2
  --h_flip_prob H_FLIP_PROB
                        probability of horizontal flip, default= 0.2
  --bright_contrast_prob BRIGHT_CONTRAST_PROB
                        probability of RandomBrightnessContrast
  --b_limit B_LIMIT     brightness_limit of RandomBrightnessContrast, default=
                        0.2
  --c_limit C_LIMIT     contrast_limit of RandomBrightnessContrast, default=
                        0.2
  --elatic_prob ELATIC_PROB
                        probability of ElasticTransform, default= 0.2
  --model_save_path MODEL_SAVE_PATH
                        a path to save a model (folder)
  --model_path MODEL_PATH
                        a path to a trained model if any (.pth)
  --model_type MODEL_TYPE
                        0:Unet, 1:Unet++, 2:DeepLabV3+, 3:UViT, 4:UPViT,
                        5:ViTV3, 6:ResUNet, 7:MobileUnet
  --encoder_name ENCODER_NAME
                        choose encoder, (ex) resnet18, resnet34, resnet50 ...
                        resnet152, xception, inceptionv4 .... refer to https:/
                        /github.com/qubvel/segmentation_models.pytorch#encoder
                        s for the details
  --encoder_weights ENCODER_WEIGHTS
                        imagenet/ssel/swsl ...
  --in_channels IN_CHANNELS
                        model input channels (1 for gray-scale images, 3 for
                        RGB, etc.)
  --classes CLASSES     model output channels (number of classes in your
                        dataset)
  --max_epoches MAX_EPOCHES
                        max number of epoches
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               learning rate
  --print_log_option PRINT_LOG_OPTION
                        print batch loss every 'print_log_option' step
  --patience PATIENCE   patience of early stopping
  --delta DELTA         delta: significant change of early stopping condition
  --best_top_k BEST_TOP_K
                        the number of models to save during training
```

Model structure:
![image](https://user-images.githubusercontent.com/51608554/170903969-2c621c04-d42d-4f4a-93e6-6dbfa5e479d6.png)
     
Experiment results
![image](https://user-images.githubusercontent.com/51608554/170904065-80d1fde0-0a52-42da-bf26-25ae4a4ae75b.png)


