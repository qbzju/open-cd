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
    def __init__(self, in_channels, patch_size=4):
        super().__init__()
        self.gap_sigmoid = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        self.gate = nn.Softmax(dim=1)

        self.norms = nn.ModuleList([
            nn.BatchNorm2d(c) for c in in_channels
        ])
        
        self.channel_attentions = nn.ModuleList([
            ChannelAttention(c) for c in in_channels
        ])


    def forward(self, xA, xB):
        """
        xA : list[Tensor] len(xA) = len(xB) = n_layers \n
        Fuse the features from different layers by concat-add multi-scale feature
        """
        assert len(xA) == len(xB), "The features xA and xB from the" \
            "backbone should be of equal length"

        outs = []
        for i in range(len(xA)):
            diff = torch.abs(xA[i] - xB[i])
            sum = xA[i] + xB[i]
            out = self.channel_attentions[i](diff) * diff
            gate = self.gate(self.gap_sigmoid(sum)) # [B, C, 1, 1]
            out = out * gate # [B, C, H, W]

            out = self.norms[i](out)
            outs.append(out)

        return outs

