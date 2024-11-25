import torch
import torch.nn as nn
from opencd.registry import MODELS
from ..backbones.focal_modulation import FocalModulation, Aggregator
from einops.layers.torch import Rearrange

class CrossFocalKernel(nn.Module):
    def __init__(self, dim, kernel_size, use_gelu=True):
        super().__init__()
        self.conv = nn.Sequential(
            Rearrange('b (g d) h w -> b (d g) h w', g=2), 
            nn.Conv2d(2 * dim, dim, kernel_size=kernel_size, 
                              stride=1, groups=2, padding=kernel_size//2, 
                              bias=False)
        )
        self.act = nn.GELU() if use_gelu else nn.Identity()
        self.H = None
        self.W = None

    def forward(self, x):
        B, L, C = x.shape # C = 2 * dim
        assert L == self.H * self.W, "input feature has wrong size"


        x_out = x.view(x.shape[0], self.H, self.W, x.shape[2]).permute(0, 3, 1, 2).contiguous() # [B, 2*C, H, W]
        x_out = self.act(self.conv(x_out))
        return x_out.view(B, C // 2, L).permute(0, 2, 1).contiguous() # [B, L, C]


class CrossFocalModulation(FocalModulation):
    # In the CrossFocalLayer layer, we do not perform multi-scale fusion,
    # but only the same-scale fusion of the two images.
    def __init__(self, dim,
                 focal_factor=2,
                 focal_level=2,
                 focal_window=7,
                 drop=0.):
        super().__init__(dim,
                         focal_factor=focal_factor,
                         focal_level=focal_level,
                         focal_window=focal_window,
                        )
        self.cross_gate_sm = nn.Softmax(dim=2)
        self.aggregator = Aggregator(dim, focal_level,kernel=CrossFocalKernel)
        self.proj = nn.Linear(dim, dim) 
        self.proj_drop = nn.Dropout(drop)
        
    def forward(self, xa, xb):
        B, C, H, W = xa.shape
        xa = xa.view(B, C, H*W).permute(0, 2, 1).contiguous()
        xb = xb.view(B, C, H*W).permute(0, 2, 1).contiguous()

        _, ctx_a, gates_a = self.modulator(xa)
        q_b, ctx_b, gates_b = self.modulator(xb)
        gates = self.cross_gate_sm(gates_a + gates_b)
        ctx_all = self.aggregator(torch.cat([ctx_a, ctx_b], dim=2), gates, H, W)
        
        x_out = q_b * self.proj_drop(self.proj(ctx_all))

        x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x_out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


@MODELS.register_module()
class FocalFusion(nn.Module):
    def __init__(self, in_channels,
                 focal_factor=2,
                 focal_level=[2,2,2,2],
                 focal_window=[7,7,7,7],
                 drop=0.):
        super().__init__()

        self.cross_focals = nn.ModuleList([
            CrossFocalModulation(c, 
                                 focal_factor, 
                                 focal_level[i], 
                                 focal_window[i], 
                                 drop) 
            for i, c in enumerate(in_channels)
        ])
        # attention for each scale
        self.channel_attentions = nn.ModuleList([
            ChannelAttention(c) for c in in_channels
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
            diff = self.base_forward(xa_i, xb_i, i)
            gate = self.gate(self.gap_sigmoid(diff))  # [B, C, 1, 1]
            out = diff * gate  # [B, C, H, W]
            outs.append(out)
        return outs
