import torch
import torch.nn as nn
from opencd.registry import MODELS
from ..backbones.focal_modulation import Aggregator, FocalKernel, Modulator
from einops.layers.torch import Rearrange


class CrossFocalKernel(nn.Module):
    def __init__(self, dim, kernel_size, use_gelu=True):
        super().__init__()
        # dim = 2 * c
        self.group_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            Rearrange('b (g d) h w -> b (d g) h w', g=2),
            nn.Conv2d(dim, dim // 2, kernel_size=kernel_size,
                      stride=1, groups=dim // 2, padding=kernel_size//2,
                      bias=False),
        )

        self.act = nn.GELU() if use_gelu else nn.Identity()
        self.bn = nn.BatchNorm2d(dim // 2)

    def forward(self, x):
        B, L, C = x.shape  # C = 2 * c
        assert L == self.H * self.W, "input feature has wrong size"

        x_out = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()  # [B, 2*c, H, W]
        x_out = self.act(self.group_conv(x_out))  # [B, C, H, W]
        return x_out.permute(0, 2, 3, 1).view(B, L, -1).contiguous()


class CrossAggregator(Aggregator):
    def __init__(self, dim, focal_level, focal_window, kernel=CrossFocalKernel, normalize_context=False):
        super().__init__(dim, focal_level, kernel, normalize_context, focal_window=focal_window)
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


class CrossFocalModulation(nn.Module):
    # In the CrossFocalLayer layer, we do not perform multi-scale fusion,
    # but only the same-scale fusion of the two images.
    def __init__(self, dim,
                 num_head=4,
                 focal_factor=2,
                 focal_level=2,
                 focal_window=7,
                 normalize_context=False,
                 drop=0.,
                 fuse_mode='concat'):
        super().__init__()
        self.num_head = num_head
        self.head_dim = dim // (2 * num_head)
        self.input_ln = nn.LayerNorm(dim // 2)
        self.fuse_mode = fuse_mode

        self.modulator = Modulator(dim // 2, focal_level, num_head=num_head)
        self.aggregator_a  = Aggregator(dim // 2, focal_level, kernel=FocalKernel, normalize_context=normalize_context, focal_window=focal_window, drop=drop, num_head=num_head  )
        self.aggregator_b  = Aggregator(dim // 2, focal_level, kernel=FocalKernel, normalize_context=normalize_context, focal_window=focal_window, drop=drop, num_head=num_head)
        self.aggregator_ab = Aggregator(dim // 2, focal_level, kernel=FocalKernel, normalize_context=normalize_context, focal_window=focal_window, drop=drop, num_head=num_head)
        self.aggregator_ba = Aggregator(dim // 2, focal_level, kernel=FocalKernel, normalize_context=normalize_context, focal_window=focal_window, drop=drop, num_head=num_head)
         
        
        if self.fuse_mode == 'concat':
            self.proj = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)

        elif self.fuse_mode == 'add':
            # TODO: Position-Wise Fusion
            self.ctx_fusion = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim // 2),
            )
            self.proj = nn.Linear(dim // 2, dim // 2)
            self.ln = nn.LayerNorm(dim // 2)

        else:
            raise ValueError(f"fuse_mode {self.fuse_mode} is not supported")
        
        self.proj_drop = nn.Dropout(drop)
    
    def forward(self, xa, xb):
        B, C, H, W = xa.shape

        short_cut_a = xa
        short_cut_b = xb

        xa = xa.view(B, C, H*W).permute(0, 2, 1).contiguous() # [B, H*W, C]
        xb = xb.view(B, C, H*W).permute(0, 2, 1).contiguous() # [B, H*W, C]

        xa = self.input_ln(xa) # [B, H*W, C]
        xb = self.input_ln(xb) # [B, H*W, C]
        
        outputs_out = []
            
        q_a, ctx_a, gates_a = self.modulator(xa)
        q_b, ctx_b, gates_b = self.modulator(xb)

        ctx_all_a = self.aggregator_a(ctx_a, gates_a, H, W)
        ctx_all_b = self.aggregator_b(ctx_b, gates_b, H, W)
        ctx_all_ab = self.aggregator_ab(ctx_a, gates_b, H, W)
        ctx_all_ba = self.aggregator_ba(ctx_b, gates_a, H, W)

        xa_out = ctx_all_a * q_a
        xb_out = ctx_all_b * q_b
        
        ctx_all_A = xa_out + ctx_all_ab * q_b
        ctx_all_B = xb_out + ctx_all_ba * q_a
        

        if self.fuse_mode == 'concat':
            outputs_a = xa_out
            outputs_b = xb_out
            outputs_out = torch.cat([ctx_all_A, ctx_all_B], dim=2)
            
        elif self.fuse_mode == 'add':
            outputs_out = self.ctx_fusion(torch.cat([ctx_all_A, ctx_all_B], dim=2))
        
        x_out = self.proj_drop(self.proj(self.ln(outputs_out.reshape(B, H*W, -1))))
        x_out = x_out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        if self.fuse_mode == 'concat':
            xa_final = outputs_a.reshape(B, H, W, C).permute(0, 3, 1, 2) + short_cut_a
            xb_final = outputs_b.reshape(B, H, W, C).permute(0, 3, 1, 2) + short_cut_b
            
            return xa_final, xb_final, x_out
        
        return None, None, x_out
    
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
        return self.sigmod(out) + 1


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
