from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_

class UpSample(nn.Sequential):
    
    def __init__(self, 
                 in_channels,
                 out_channels,
                 groups 
                ):
        super().__init__(
            nn.ConvTranspose2d(in_channels, in_channels, 2, 2, groups= groups, bias= False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ConvTrans2dBlock(nn.Sequential):

    def __init__(self, in_channels,
                       out_channels):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ) 

class DepthwiseSeparableConv2d(nn.Module):

    def __init__(self, in_chans, out_chans, kernel_size, padding, dilation_rate, stride= 1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size, stride, 
                        padding, dilation_rate, groups= in_chans),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, out_chans, 1, stride),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
        )
        self.mapping = nn.Conv2d(in_chans, out_chans, 1)

    def forward(self, x):
        fx = self.f(x)
        x = F.relu(fx+self.mapping(x), inplace=True)
        return x                

class DoubleConvResidBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConvResidBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace= True),
                nn.Conv2d(out_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels)
        )
        self.mapping = nn.Conv2d(in_channels, out_channels, (1,1))
        
    def forward(self, x):
        out = self.block(x)
        out = out + self.mapping(x)
        out = F.relu(out, inplace= True)
        return out

class DoubleLightConvResidBlock(nn.Module):

    def __init__(self, in_chans, out_chans, kernel_size, padding, dilation_rate, stride= 1):
        super(DoubleLightConvResidBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size, stride, 
                        padding, dilation_rate, groups= in_chans),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, out_chans, 1, stride),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size, stride, 
                        padding, dilation_rate, groups= out_chans),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, 1, stride),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()            
        )
        self.mapping = nn.Conv2d(in_chans, out_chans, (1,1))
        
    def forward(self, x):
        out = self.f(x)
        out = out + self.mapping(x)
        out = F.relu(out, inplace= True)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features:int, 
                        hidden_features=None, 
                        out_features=None, 
                        act_layer=nn.GELU, 
                        drop:float =0.):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialReductionAttention(nn.Module):
    '''Spatial reduction attention,
       ordinary attention block when sr_ratio = 1
    '''
    def __init__(self, dim, num_heads= 8, 
                 qkv_bias = False, qk_scale= None,
                 attn_drop= 0., proj_drop= 0., sr_ratio= 1,
                 is_DSC= True
                ):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim: {dim} should be divisible by num_heads: {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        # embedding is equally distributed to each head.
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias= qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias= qkv_bias)
        
        self.proj = nn.Linear(dim, dim) # project attention@value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if is_DSC\
                else DepthwiseSeparableConv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape # B HW C(dim)
        
        # create query vectors
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        # B, N, C -> B, N, dim -> B, num_heads, N, dim//num_heads

        # create key, value vectors
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C//self.num_heads)
            kv = kv.permute(2, 0, 3, 1, 4)
            # B, N, C -> B, N, dim*2 
            # -> B, N, num_heads, dim*2//num_heads
            # -> B, N, num_heads, 2 dim//num_heads
            # -> 2, B, num_heads, N, dim//num_heads
        k, v = kv[0], kv[1]
        # B, num_heads, N, dim//num_heads
        
        attn = F.softmax(q@k.transpose(-1,-2)*self.scale, dim= -1)
        # B, num_heads, N, N
        attn= self.attn_drop(attn)
        
        x = attn@v
        # B, num_heads, N, dim//num_heads
        x = x.permute(0, 2, 1, 3).reshape(B, N, C) # B, N, C
        x = self.proj(x) # B, N, C
        x = self.proj_drop(x)

        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(self, 
                 dim: int, num_heads: int,
                 mlp_ratio: int = 4, 
                 qkv_bias: bool = False, 
                 qk_scale = None, 
                 proj_drop: float = 0.,
                 attn_drop: float = 0., 
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm, 
                 sr_ratio: int = 1,
                 is_DSC: bool = True
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialReductionAttention(
            dim= dim,
            num_heads= num_heads,
            qkv_bias= qkv_bias,
            qk_scale= qk_scale,
            attn_drop= attn_drop,
            proj_drop= proj_drop,
            sr_ratio= sr_ratio,
            is_DSC= is_DSC
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features= dim, hidden_features= mlp_hidden_dim, 
                        out_features= dim,
                        act_layer= act_layer, drop= proj_drop)
    
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    '''Image to patch embedding
    '''
    def __init__(self, 
                 img_size:int = 224, 
                 patch_size:int = 16, 
                 in_chans:int = 3,
                 dim:int = 768
                ):
        super().__init__()
        img_size = to_2tuple(img_size) # (img_size, img_size)
        patch_size = to_2tuple(patch_size) # (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, dim, 
                              kernel_size= patch_size, 
                              stride= patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, return_shape= True):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(-1,-2)
        # B, C, H, W -> B, dim, H//P, W//P
        # -> B, dim HW//P^2 -> B, HW//P^2 dim
        x = self.norm(x)
        H, W = H//self.patch_size[0], W//self.patch_size[1]
        if return_shape:
            return x, (H, W)
        else:
            return x

class PyramidVisionTransformer(nn.Module):

    def __init__(self, img_size= 224, patch_size= 16, in_chans= 3,
                 num_classes= 1000, embed_dims=[64, 128, 256, 512],
                 num_heads= [1, 2, 4, 8], mlp_ratio= [4, 4, 4, 4],
                 qkv_bias= False, qk_scale=None, 
                 drop_rate= 0., attn_drop_rate= 0,
                 norm_layer= nn.LayerNorm, 
                 depths= [3, 4, 6, 3],
                 sr_ratios= [8, 4, 2, 1], 
                 is_DSC= True,
                 num_stages= 4
                ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths # the number of encoder blocks per transformer layer = [3, 4, 6, 3]
        self.num_stages = num_stages # the number of transformer layers = 4
        
        for i in range(num_stages):

            # patch embedding of i'th stage
            patch_embed = PatchEmbed(
                img_size= img_size if i == 0 else img_size//(2**(i+1)),
                patch_size= patch_size if i == 0 else 2,
                in_chans= in_chans if i == 0 else embed_dims[i-1],
                dim = embed_dims[i]
            )
            num_patches = patch_embed.num_patches if i != num_stages-1 else patch_embed.num_patches+1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(drop_rate)
            
            # transformer blocks
            block = nn.ModuleList(
                [TransformerEncoderBlock(
                    embed_dims[i], num_heads[i], mlp_ratio[i],
                    qkv_bias, qk_scale, drop_rate, attn_drop_rate, 
                    norm_layer= norm_layer, sr_ratio= sr_ratios[i],
                    is_DSC= is_DSC
                ) for _ in range(depths[i])]
            )

            setattr(self, f'patch_embed{i+1}', patch_embed)
            setattr(self, f'pos_embed{i+1}', pos_embed)
            setattr(self, f'pos_drop{i+1}', pos_drop)
            setattr(self, f'block{i+1}', block)

            trunc_normal_(pos_embed, std=0.02)
        
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=True).reshape(1, -1, H * W).permute(0, 2, 1)            
    
    def forward(self, x):
        outs = []
        B = x.shape[0]

        for i  in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i+1}")
            pos_embed = getattr(self, f"pos_embed{i+1}")
            pos_drop = getattr(self, f"pos_drop{i+1}")
            block = getattr(self, f"block{i+1}")
            x, (H,W) = patch_embed(x)
            if i == self.num_stages -1:
                pos_embed = self._get_pos_embed(pos_embed[:,1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            x = pos_drop(x+pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        
        return outs

class ASPP(nn.Module):

    def __init__(self, emb_size= 768, branch_emb_size = 768//8,
                       dilation_rates= [6, 12, 24]
                ):
        super().__init__()

        # (1) 1x1 conv 
        self.branch1 = DepthwiseSeparableConv2d(emb_size, branch_emb_size, 1, 0, 1)
    
        # (2) 3x3 conv rate: 6,  padding: 6
        self.branch2 = DepthwiseSeparableConv2d(emb_size, branch_emb_size, 3, dilation_rates[0], dilation_rates[0])

        # (3) 3x3 conv rate: 12, padding: 12
        self.branch3 =  DepthwiseSeparableConv2d(emb_size, branch_emb_size, 3, dilation_rates[1], dilation_rates[1])

        # (4) 3x3 conv rate: 18, padding: 18
        self.branch4 =  DepthwiseSeparableConv2d(emb_size, branch_emb_size, 3, dilation_rates[2], dilation_rates[2])

        # (5) Image pooling: AdaptivePoold2d + 1x1 conv
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(emb_size, branch_emb_size, 1),
            nn.BatchNorm2d(branch_emb_size),
            nn.ReLU()
        )
        
        self.emb_size = emb_size
        self.branch_size = branch_emb_size
        self.dilation_rates = dilation_rates
    
    def forward(self, x):
        B, C, H, W = x.shape
        outs = []
        for i in range(1, 6):
            branch = getattr(self, f"branch{i}")
            outs.append(branch(x))
        outs[4] = F.upsample(outs[4], size= (H, W),
                                mode= 'bilinear', align_corners=True)
        return outs


# if __name__ == '__main__':
#     x = torch.randn(1,256,32,32)
#     H,W = x.shape[-2:]
#     patchemb = PatchEmbed(32, 16, 256, 256)
#     patch = patchemb(x, return_shape= False)
#     block = TransformerEncoderBlock(256, 8)
#     encoded = block(patch, H, W)

#     print("patch", patch.shape)
#     print("encoded", encoded.shape)


    # aspp = ASPP(emb_size = 64, branch_emb_size= 8)
    # aspp.eval()
    # out = aspp(x)
    # for ot in out:
    #     print(ot.shape)