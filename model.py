import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from modules import *
from torchvision import models

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
                 in_channels= 3,
                 patch_size= 2,
                 img_size= 224,
                 dilation_rates = [6, 12, 24],
                 **kwargs
                ):
        
        super().__init__()
        # CNN backbone to expand
        # b, 3, 224, 224 --> b, 64, 224, 224
        self.cnn_expansion_layers = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, 4, 3, 1, 1),
            DepthwiseSeparableConv2d(4, 8, 3, 1, 1),
            DepthwiseSeparableConv2d(8, 16, 3, 1, 1),
            DepthwiseSeparableConv2d(16, 32, 3, 1, 1)
        )

        # Deepvlab3+ structure (BN+ReLU)
        self.aspp = ASPP(emb_size = 32, branch_emb_size= 4, dilation_rates= dilation_rates)
        # [(b, 4, 224, 224)] * 5

        # Pvit modules 
        for i in range(5):
            pvit = PyramidVisionTransformer(img_size= img_size, 
                                            in_chans= 4,
                                            patch_size= patch_size,
                                            embed_dims= [4, 8, 16, 32],
                                            **kwargs)
            setattr(self, f"pvit{i}", pvit)
        # model outputs.
        # b, 8, 112, 112
        # -> b, 16, 56, 56
        # -> b, 32, 28, 28
        # -> b, 64, 14, 14   

        self.cnn_dimesion_reduction_layers = nn.Sequential(
            DepthwiseSeparableConv2d(32*5, 32, 3, 1, 1),
            DepthwiseSeparableConv2d(32, 16, 3, 1, 1),
            DepthwiseSeparableConv2d(16, 8, 3, 1, 1),
            DepthwiseSeparableConv2d(8, 4, 3, 1, 1)
        )     

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(4, num_classes, 2, 2), 
            nn.Upsample(size=(56, 56) , mode='bilinear'),
            nn.Upsample(size=(112, 112) , mode='bilinear'),
            nn.Upsample(size=(224, 224) , mode='bilinear')
        )   

    def forward(self, x):
        x = self.cnn_expansion_layers(x)
        x = self.aspp(x)
        for i in range(5):
            pvit = getattr(self, f"pvit{i}")
            x[i] = pvit(x[i])[-1]
        x = torch.cat(x, dim= 1) # b, 64*5, 14, 14
        x = self.cnn_dimesion_reduction_layers(x)
        x = self.decode(x)
        return x, None

# U-net using Resnet as a backbone network   
class ResUNet(nn.Module):

    def __init__(self, in_channels, out_channels, img_size= 224):
        super(ResUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.backbone = models.resnet34(pretrained= True)
        self.base_layers = list(self.backbone.children())

        # encoder layer (resnet)
        # assume input image batch's shape = bs, 3, 224, 224
        self.enc0 = nn.Sequential(*self.base_layers[:3]) 
        self.enc1 = nn.Sequential(*self.base_layers[3:5])
        self.enc2 = nn.Sequential(*self.base_layers[5])
        self.enc3 = nn.Sequential(*self.base_layers[6])
        self.enc4 = nn.Sequential(*self.base_layers[7])

        # decoder layer
        # up-sample
        self.up1 = UpSample(512, 512, 512) 
        # concat up1 and enc3: bs, 256+256, 32, 32
        self.conv1 = DoubleConvResidBlock(256+256, 256) 
        
        self.up2 = UpSample(256, 256, 256) 
        # concat up2 and enc2: bs, 128+128, 64, 64
        self.conv2 = DoubleConvResidBlock(128+128, 128)
        
        self.up3 = UpSample(128, 128, 128) # bs, 64, 128, 128
        # concat up3 and enc1: bs, 64+64, 128, 128
        self.conv3 = DoubleConvResidBlock(64+64, 64) 

        self.up4 = UpSample(64, 64, 64) 
        # concat up4 and enc0: bs, 64+64, 128, 128
        self.conv4 = DoubleConvResidBlock(64+32, 64)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2), 
            nn.Conv2d(32, out_channels, 1)
        )
        
    def forward(self, x):
        # encoder
        c1 = self.enc0(x) 
        c2 = self.enc1(c1)
        c3 = self.enc2(c2)
        c4 = self.enc3(c3)
        c5 = self.enc4(c4)

        # decoder 
        u1 = self.up1(c5)
        cat1 = torch.cat([u1, c4], dim= 1)
        uc1 = self.conv1(cat1)  

        u2 = self.up2(uc1)
        cat2 = torch.cat([u2, c3], dim= 1)
        uc2 = self.conv2(cat2) 

        u3 = self.up3(uc2)
        cat3 = torch.cat([u3, c2], dim= 1)
        uc3 = self.conv3(cat3) 

        u4 = self.up4(uc3)
        cat4 = torch.cat([u4, c1], dim= 1)
        uc4 = self.conv4(cat4) 

        # return decoder
        out = self.decode(uc4)
        
        return out, None


# U-net using mobile-net as a backbone network   
class MobileUnet(nn.Module):

    def __init__(self, in_channels, out_channels, img_size= 224):
        super(MobileUnet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        backbone =  torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        base_layers = list(list(backbone.children())[0].children())

        # encoders
        self.enc0 = nn.Sequential(*base_layers[0:2])# b, 16, 112, 112
        self.enc1 = nn.Sequential(*base_layers[2:4])# b 24, 56, 56
        self.enc2 = nn.Sequential(*base_layers[4:7])# b, 32, 28, 28
        self.enc3 = nn.Sequential(*base_layers[7:14]) # b, 96, 14, 14
        self.enc4 = nn.Sequential(*base_layers[14:])# b, 1280, 7, 7 

        # decoders
        self.conv4 = nn.Sequential(
            DoubleLightConvResidBlock(1280, 640, 1, 0, 1, 1),
            DoubleLightConvResidBlock(640, 320, 1, 0, 1, 1),
            DoubleLightConvResidBlock(320, 192, 1, 0, 1, 1)
        )

        self.up4 = UpSample(192, 96, 192)
        self.conv3 = DoubleLightConvResidBlock(96+96, 96, 3, 1, 1, 1)
        self.up3 = UpSample(96, 32, 96)
        self.conv2 = DoubleLightConvResidBlock(32+32, 32, 3, 1, 1, 1)
        self.up2 = UpSample(32, 24, 32)
        self.conv1 = DoubleLightConvResidBlock(24+24, 24, 3, 1, 1, 1)
        self.up1 = UpSample(24, 16, 24)
        self.conv0 = DoubleLightConvResidBlock(16+16, 16, 3, 1, 1, 1)
        self.decode = nn.Sequential(
            nn.Upsample((img_size, img_size), mode= 'bilinear'),
            DepthwiseSeparableConv2d(16, out_channels, 1, 0, 1, 1)
        )
        
    def forward(self, x):
        # encoder
        c0 = self.enc0(x) # b, 16, 112, 112
        c1 = self.enc1(c0) # b 24, 56, 56
        c2 = self.enc2(c1) # b, 32, 28, 28
        c3 = self.enc3(c2) # b, 96, 14, 14
        c4 = self.enc4(c3) # b, 1280, 7, 7 

        # decoders
        c4 = self.conv4(c4)
        c4 = self.up4(c4)

        c3 = torch.cat([c3, c4], dim= 1)
        c3 = self.conv3(c3)
        c3 = self.up3(c3)

        c2 = torch.cat([c2,c3], dim= 1)
        c2 = self.conv2(c2)
        c2 = self.up2(c2)

        c1 = torch.cat([c1, c2], dim= 1)
        c1 = self.conv1(c1)
        c1 = self.up1(c1)

        c0 = torch.cat([c0, c1], dim= 1)
        c0 = self.conv0(c0)
        c0 = self.decode(c0)

        return c0, None

# if __name__ == '__main__':
#     x = torch.randn(2, 3, 224, 224)
#     model = LightResUNet(3, 1)
#     model.eval()
#     out, _ = model(x)
#     print(out.shape)
