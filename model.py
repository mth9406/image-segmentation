import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from modules import *

class MyViT(nn.Module):

    def __init__(self, 
                 num_classes,
                 model_url= "google/vit-base-patch16-224-in21k",
                 patch_size= 16,
                 img_size= 224,
                 emb_size= 768
                ):
        
        super().__init__()
        net = ViTModel.from_pretrained(model_url)
        backbone = list(net.children())

        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_url)
        self.vit_embedding = backbone[0]
        self.vit_layers = list(backbone[1].modules())[1]
        # self.layer_norm = backbone[2]
        # self.vit_pooler = backbone[3]
        self.up1_1 = UpSample(emb_size*3, emb_size, groups=3)
        self.up1_2 = UpSample(emb_size*3, emb_size, groups=3)
        self.up1_3 = UpSample(emb_size*3, emb_size, groups=3)
        self.up1_4 = UpSample(emb_size*3, emb_size, groups=3)    

        self.up2_1 = UpSample(emb_size, emb_size//3, groups=2)
        self.up2_2 = UpSample(emb_size, emb_size//3, groups=2)

        self.up3 = UpSample(emb_size//3, emb_size//6, groups= 1)
        self.up4 = UpSample(emb_size//12,emb_size//24, groups= 1)
        self.fc = nn.Conv2d(emb_size//48, num_classes, 1)

        self.patch_size = patch_size
        self.img_size = img_size
        self.cls_size = 1
        self.num_patches = img_size//patch_size # 14
        self.emb_size= 768


    def forward(self, x):
        # x = self.feature_extractor(x)
        x = self.vit_embedding(x)
        outs= []
        for vit_layer in self.vit_layers:
            x = vit_layer(x)[0]
            outs.append(self._patchify(x, self.num_patches))
        # outs = torch.cat(outs, dim= 1) # b, 768*12 14 14
        # outs = self.upsample(outs)
        out1, out2, out3, out4 = torch.cat(outs[:3], dim=1), torch.cat(outs[3:6], dim=1), torch.cat(outs[6:9], dim=1), torch.cat(outs[9:], dim=1)
        del outs
        out1, out2, out3, out4 = self.up1_1(out1), self.up1_2(out2), self.up1_3(out3), self.up1_4(out4)

        out1_2 = torch.cat([out1, out2], dim= 1)
        del out1, out2
        out1_2 = self.up2_1(out1_2)
        out3_4 = torch.cat([out3, out4], dim= 1)
        del out3, out4
        out3_4 = self.up2_2(out3_4)
        
        outs = torch.cat([out1_2, out3_4], dim= 1)
        del out1_2, out3_4
        outs = self.up3(outs)
        outs = self.up4(outs)
        outs = self.fc(outs)

        return outs, None


    def _patchify(self, x, num_patches):
        cls_token, patches = torch.split(x, [1, num_patches**2], dim= 1)
        patches = rearrange(patches, 'b (s1 s2) e -> b e s1 s2', s1= num_patches, s2= num_patches)
        # b 768 14 14
        return patches

class UPyramidVisionTransformer(nn.Module):
    
    def __init__(self, num_classes, img_size= 224, **config):
        
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Encoder
        self.backbone = PyramidVisionTransformer(img_size= img_size, 
                                                 patch_size= 4,
                                                 embed_dims= [64, 128, 256, 512],
                                                 **config)
        # default model outputs.
        # b, 64, 14, 14
        # b, 128, 7, 7
        # b, 256, 3, 3
        # b, 512, 1, 1

        # when patch_size = 2,
        # embed_dim = [64, 128, 256, 512]
        # b, 64, 112, 112
        # b, 128, 56, 56
        # b, 256, 28, 2800
        # b, 512, 14, 14

        # Decoder
        for i in range(9,6,-1):
            up = ConvTrans2dBlock(2**i, 2**(i-1))
            fc = nn.Conv2d(2**i, 2**(i-1),1, groups= 2)
            setattr(self, f"up{10-i}", up)
            setattr(self, f"fc{10-i}", fc)
        self.up4 = ConvTrans2dBlock(64, 32)
        self.decode = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding= 1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace= True),
            ConvTrans2dBlock(16, 8),
            nn.Conv2d(8, self.num_classes, 1)
        )
         
    def forward(self, x):
        outs = self.backbone(x)
        outs.reverse()
        for i in range(3):
            up, fc = getattr(self, f"up{i+1}"), getattr(self, f"fc{i+1}")
            outs[i] = up(outs[i])
            outs[i+1] = torch.cat([outs[i+1], outs[i]], dim=1)
            outs[i+1] = fc(outs[i+1])
        outs = self.up4(outs[3]) 
        outs = self.decode(outs)
        return outs, None

class ViTV3(nn.Module):

    def __init__(self, 
                 num_classes,
                 model_url= "google/vit-base-patch16-224-in21k",
                 patch_size= 16,
                 img_size= 224,
                 emb_size= 768
                ):
        
        super().__init__()
        net = ViTModel.from_pretrained(model_url)
        backbone = list(net.children())

        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_url)
        self.vit_embedding = backbone[0]
        self.vit_layers = list(backbone[1].modules())[1]
        self.layer_norm = backbone[2]
        # self.vit_pooler = backbone[3]

        # Deepvlab3+ structure (BN+ReLU)
        num_patches = img_size//patch_size # 14
        branch_emb_size = emb_size//8
        # (1) 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(emb_size, branch_emb_size, 1),
            nn.BatchNorm2d(branch_emb_size),
            nn.ReLU()
        )
        # (2) 3x3 conv rate: 6,  padding: 6
        self.branch2 = nn.Sequential(
            nn.Conv2d(emb_size, branch_emb_size, 3, 1, 6, 6),
            nn.BatchNorm2d(branch_emb_size),
            nn.ReLU()
        )
        # (3) 3x3 conv rate: 12, padding: 12
        self.branch3 = nn.Sequential(
            nn.Conv2d(emb_size, branch_emb_size, 3, 1, 12, 12),
            nn.BatchNorm2d(branch_emb_size),
            nn.ReLU()
        )
        # (4) 3x3 conv rate: 18, padding: 18
        self.branch4 = nn.Sequential(
            nn.Conv2d(emb_size, branch_emb_size, 3, 1, 18, 18),
            nn.BatchNorm2d(branch_emb_size),
            nn.ReLU()
        )
        # (5) Image pooling: AdaptivePoold2d + 1x1 conv
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(emb_size, branch_emb_size, 1),
            nn.BatchNorm2d(branch_emb_size),
            nn.ReLU()
        )

        # (6) = concat((1) ~ (5))
        # b, 480, 14, 14
        branches_emb_size = branch_emb_size * 5
        # upsample (bilinear) (6) 4 times (14,14 -> 224, 224)
        self.decode = nn.Sequential(
            nn.Conv2d(branches_emb_size, branch_emb_size, 1),
            nn.BatchNorm2d(branch_emb_size),
            nn.Conv2d(branch_emb_size, num_classes, 1)
        )

        self.patch_size = patch_size
        self.img_size = img_size
        self.cls_size = 1
        self.num_patches = num_patches # 14
        self.emb_size= 768


    def forward(self, x):
        # x = self.feature_extractor(x)
        x = self.vit_embedding(x)
        for vit_layer in self.vit_layers:
            x = vit_layer(x)[0]
        x = self.layer_norm(x) # b, 768, 1+14**2
        x = self._patchify(x, num_patches= self.num_patches) # b, 768, 14, 14

        # Deepvlab3+ structure
        outs = []
        for i in range(1, 6):
            branch = getattr(self, f"branch{i}")
            outs.append(branch(x))
        outs[4] = F.upsample(outs[4], size= (self.num_patches, self.num_patches), 
                             mode= 'bilinear', align_corners= False)

        outs = torch.cat(outs, dim=1)
        outs = self.decode(outs) # b, num_classes, patch_size, patch_size
        outs = F.upsample(outs, size=(self.img_size, self.img_size), 
                             mode='bilinear', align_corners=False)

        return outs, None

    def _patchify(self, x, num_patches):
        cls_token, patches = torch.split(x, [1, num_patches**2], dim= 1)
        patches = rearrange(patches, 'b (s1 s2) e -> b e s1 s2', s1= num_patches, s2= num_patches)
        # b 768 14 14
        return patches

# x = torch.randn(2, 3, 224, 224)
# upvit = ViTV3(1)
# upvit.eval()
# out, _ = upvit(x)
# print(out.shape)
