import torch
import torch.nn as nn
from opencd.registry import MODELS

@MODELS.register_module()
class FocalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap_sigmoid = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        self.gate = nn.Softmax(dim=1)

    def forward(self, x):

        outs = []
        for i in range(len(x)):
            gate = self.gate(self.gap_sigmoid(x[i])) # [B, 2C, 1, 1]
            x[i] = x[i] * gate # [B, 2C, H, W]
            outs.append(x[i].sum(dim=1))

        return (outs, )