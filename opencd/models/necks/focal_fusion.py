import torch
import torch.nn as nn
from opencd.registry import MODELS
from ..backbones.focal_modulation import FocalModulation, Aggregator
from einops.layers.torch import Rearrange


class CrossFocalKernel(nn.Module):
    def __init__(self, dim, kernel_size, use_gelu=True):
        super().__init__()
        # dim = 2 * c
        self.group_conv = nn.Sequential(
            Rearrange('b (g d) h w -> b (d g) h w', g=2),
            nn.Conv2d(dim, dim // 2, kernel_size=kernel_size,
                      stride=1, groups=dim // 2, padding=kernel_size//2,
                      bias=False),
        )

        self.act = nn.GELU() if use_gelu else nn.Identity()
        self.H = None
        self.W = None

    def forward(self, x):
        B, L, C = x.shape  # C = 2 * c
        assert L == self.H * self.W, "input feature has wrong size"

        x_out = x.view(B, self.H, self.W, C).permute(
            0, 3, 1, 2).contiguous()  # [B, 2*c, H, W]
        x_out = self.act(self.group_conv(x_out))  # [B, C, H, W]
        return x_out.permute(0, 2, 3, 1).view(B, L, -1).contiguous()


class CrossAggregator(Aggregator):
    def __init__(self, dim, focal_level, kernel=CrossFocalKernel):
        super().__init__(dim, focal_level, kernel)
        self.h = nn.Conv2d(dim // 2, dim // 2, 1, 1, 0, groups=1, bias=True)

    def forward(self, context, gates, H=None, W=None):

        B, L, C = context.shape
        assert L == H * W, "input feature has wrong size"
        ctx_all = 0
        ctx = context
        for i, layer in enumerate(self.layers):
            layer.H, layer.W = H, W
            ctx_i = layer(ctx)
            ctx_all = ctx_all + ctx_i * gates[:, :, i:i+1]

        ctx_global = self.act(ctx_all.mean(dim=1, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, :, self.depth:]

        if self.normalize_context:
            ctx_all = ctx_all / (self.depth + 1)

        ctx_all = ctx_all.view(B, H, W, -1).permute(0, 3,
                                                    # [B, C, H, W]
                                                    1, 2).contiguous()
        ctx_all = self.h(ctx_all)
        # [B, L, C]
        return ctx_all.view(B, -1, L).permute(0, 2, 1).contiguous()


class CrossFocalModulation(FocalModulation):
    # In the CrossFocalLayer layer, we do not perform multi-scale fusion,
    # but only the same-scale fusion of the two images.
    def __init__(self, dim,
                 focal_factor=2,
                 focal_level=2,
                 focal_window=7,
                 drop=0.):
        super().__init__(dim // 2,
                         focal_factor=focal_factor,
                         focal_level=focal_level,
                         focal_window=focal_window,
                         )
        self.aggregator = CrossAggregator(
            dim, focal_level, kernel=CrossFocalKernel)
        self.proj = nn.Linear(dim // 2, dim // 2)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, xa, xb):
        B, C, H, W = xa.shape
        xa = xa.view(B, C, H*W).permute(0, 2, 1).contiguous()
        xb = xb.view(B, C, H*W).permute(0, 2, 1).contiguous()

        _, ctx_a, gates_a = self.modulator(xa)
        q_b, ctx_b, gates_b = self.modulator(xb)
        gates = gates_a + gates_b
        ctx_all = self.aggregator(
            torch.cat([ctx_a, ctx_b], dim=2), gates, H, W)  # B, L, 2c

        ctx_all *= q_b
        x_out = self.proj_drop(self.proj(ctx_all))

        return x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


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
                 focal_level=[2, 2, 2, 2],
                 focal_window=[7, 7, 7, 7],
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
            ChannelAttention(c // 2) for c in in_channels
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
