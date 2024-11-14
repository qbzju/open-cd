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

    def forward(self, xA, xB):
        
        assert len(xA) == len(xB), "The features xA and xB from the" \
            "backbone should be of equal length"

        outs = []
        for i in range(len(xA)):
            out = xA[i] + xB[i]
            gate = self.gate(self.gap_sigmoid(out)) # [B, C, 1, 1]
            out = out * gate # [B, C, H, W]
            outs.append(out)

        return outs