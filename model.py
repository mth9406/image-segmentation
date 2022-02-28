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
                 out_channels 
                ):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels)
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
        # self.conv2ds = [nn.Conv2d(emb_size, emb_size//12, 3, padding=1) for _ in range(12)]
        self.upsample = nn.Sequential(
                    nn.Conv2d(emb_size*12, emb_size, 3, padding= 1, groups= 12),
                    UpSample(emb_size, emb_size//4),
                    nn.ReLU(inplace= True),
                    UpSample(emb_size//4, emb_size//16),
                    nn.ReLU(inplace= True),
                    UpSample(emb_size//16, emb_size//64),
                    nn.ReLU(inplace= True),
                    UpSample(emb_size//64, num_classes),
                    nn.Tanh()
                )


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
        # for i, out in enumerate(outs):
        #     outs[i] = self.conv2ds[i](out) # b, 64, 14, 14
        outs = torch.cat(outs, dim= 1) # b 768*12 14 14 
        outs = self.upsample(outs)
        return outs, None


    def _patchify(self, x, num_patches):
        cls_token, patches = torch.split(x, [1, num_patches**2], dim= 1)
        patches = rearrange(patches, 'b (s1 s2) e -> b e s1 s2', s1= num_patches, s2= num_patches)
        # b 768 14 14
        return patches

input = torch.randn(1, 3, 224, 224)
vit = MyViT(1)
output, _ = vit(input)

print(output.shape)
