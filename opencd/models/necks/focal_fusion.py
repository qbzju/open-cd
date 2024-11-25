import torch
import torch.nn as nn
from opencd.registry import MODELS
from ..backbones.focal_modulation import BasicLayer

class CrossFocal(nn.Module):
    # In the CrossFocalLayer layer, we do not perform multi-scale fusion, 
    # but only the same-scale fusion of the two images.
    def __init__(self, dim, 
                 depth = 2,
                 focal_level=2,
                 focal_window=9,
                 drop=0.):
        super().__init__()
        self.cf_layer = BasicLayer(2*dim, 
                                   depth=depth, 
                                   focal_level=focal_level, 
                                   focal_window=focal_window,
                                   drop=drop)
        

    def forward(self, xa, xb):
        B, C, H, W = xa.shape
        x = torch.cat([xa, xb], dim=1)
        x = x.flatten(2).transpose(1, 2)
        x_out, _, _, _, _, _ = self.cf_layer(x, H, W)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
    

@MODELS.register_module()
class FocalFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.cross_focals = nn.ModuleList([
            CrossFocal(c) for c in in_channels
        ])
        # attention for each scale
        self.channel_attentions = nn.ModuleList([
            ChannelAttention(2*c) for c in in_channels
        ])

        self.gap_sigmoid = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        self.gate = nn.Softmax(dim=1)

    def base_forward(self, xa, xb, i):
        x = self.cross_focals[i](xa, xb)

        return self.channel_attentions[i](x) * x

    def forward(self, xA, xB):
        """
        xA : list[Tensor] len(xA) = len(xB) = n_layers \n
        Fuse the features from different layers by concat-add multi-scale feature
        """
        assert len(xA) == len(xB), "The features xA and xB from the" \
            "backbone should be of equal length"

        outs = []
        for i, (xa_i, xb_i) in enumerate(zip(xA, xB)):   
            # diff = self.base_forward(xa_i, xb_i, i)
            diff = xa_i - xb_i
            gate = self.gate(self.gap_sigmoid(diff)) # [B, C, 1, 1]
            out = diff * gate # [B, C, H, W]
            outs.append(out)
        return outs
