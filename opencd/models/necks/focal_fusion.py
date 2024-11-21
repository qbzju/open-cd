import torch
import torch.nn as nn
from opencd.registry import MODELS


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
        self.channel_attentions = nn.ModuleList([
            ChannelAttention(c) for c in in_channels
        ])

    def base_forward(self, xa, xb, i):
        xa = self.channel_attentions[i](xa) * xa
        xb = self.channel_attentions[i](xb) * xb
        return xa, xb

    def forward(self, xA, xB):
        """
        xA : list[Tensor] len(xA) = len(xB) = n_layers \n
        Fuse the features from different layers by concat-add multi-scale feature
        """
        assert len(xA) == len(xB), "The features xA and xB from the" \
            "backbone should be of equal length"

        outs = []
        for i in range(len(xA)):   
            xa, xb = self.base_forward(xA[i], xB[i], i)
            outs.append(xa + xb)

        return outs
