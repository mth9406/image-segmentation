import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class UpSample(nn.Sequential):
    
    def __init__(self, 
                 in_channels,
                 out_channels,
                 groups 
                ):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, groups= groups, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels//2, 1, bias= False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
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

x = torch.randn(1, 3, 224, 224)
vit = MyViT(1)
out, _ = vit(x)
print(out.shape)