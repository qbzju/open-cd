import torch
import torch.nn as nn
from opencd.registry import MODELS



class CrossFocalBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                 focal_level=2, focal_window=9, use_layerscale=False, layerscale_value=1e-4, normalize_modulator=True,
                 use_postln=False, use_postln_in_modulation=False):
        super().__init__()

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
        )

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.gate_sm = nn.Softmax(dim=1)
        self.focal_layers = nn.ModuleList()

    def forward(self, xa, xb):
        xb = self.f(xb)
        q, ctx, gates = torch.split(xb, 
            (self.in_channels, self.in_channels, self.focal_level+1), 
            dim=1)
            
        gates = self.gate(gates)
        
        ctx_all = 0
        for l in range(self.focal_level):
            ctx_l = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx_l * gates[:,l:l+1]
            
        ctx_global = self.act(ctx_all.mean(dim=(2,3), keepdim=True))
        ctx_all = (ctx_all + ctx_global * gates[:,self.focal_level:]) / (self.focal_level + 1)
        
        out = xa * self.h(ctx_all)
        
        out = self.proj(out)
        return out


class CrossFocalLayer(nn.Module):
    # In the CrossFocalLayer layer, we do not perform multi-scale fusion, 
    # but only the same-scale fusion of the two images.
    def __init__(self, dim, 
                 focal_level=[2, 2, 2, 2],
                 focal_window=[9, 9, 9, 9],
                 depths=[2, 2, 6, 2],
                 drop=0., drop_path=0.):
        super().__init__()
        self.dim = 2*dim
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.depths = depths

        # TODO: simplify cross-focal-block
        self.f = nn.Linear(self.dim, 2*self.dim+(self.depth+1), bias=True)
        self.act = nn.GELU()
        self.h = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(drop)

        self.gate_sm = nn.Softmax(dim=2)
        self.norm = nn.LayerNorm(self.dim)

        self.cross_focals = nn.ModuleList([
            CrossFocalBlock(2 * dim, mlp_ratio=4., drop=0., drop_path=0., 
                           focal_level=focal_level[i], focal_window=focal_window[i], use_layerscale=False, layerscale_value=1/torch.sqrt(torch.tensor(2)), normalize_modulator=True,
                           use_postln=False, use_postln_in_modulation=False) for i in range(len(depths))
                           ])

    def forward(self, xa, xb):
        # TODO: Implement cross-focal modulation
        B, C, H, W = xa.shape
        L = H * W
        x = torch.cat([xa, xb], dim=1).permute(0, 2, 3, 1).view(B, L, 2*C)
        
        assert x.shape[2] == self.dim, "The features x should have the same dimension as the input"
        x = self.norm(x)

        x = self.f(x)
        q, ctx, gates = torch.split(x, (C, C, self.depth+1), 2)  # B L, (C, C, self.depth+1)
        # use softmax to normalize the gates
        gates = self.gate_sm(gates)

        ctx_all = 0
        for i, blk in enumerate(self.cross_focals):
            ctx = blk(ctx)
            ctx_all = ctx_all + ctx * gates[:, :, i:i+1]

        ctx_global = self.act(ctx_all.mean(dim=1, keepdim=True))
        ctx_all = (ctx_all + ctx_global * gates[:, :, self.depth:]) / (self.depth + 1)

        # if self.normalize_layer:
        #     ctx_all = ctx_all / (self.depth + 1)

        # TODO: convert after multiply
        q = q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_out = q * self.h(ctx_all.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        # TODO: add ln
        # if self.use_postln:
        #     x = self.ln(x)

        x_out = x_out.permute(0, 2, 3, 1).view(B, L, C).contiguous()
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


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
            CrossFocalLayer(c) for c in in_channels
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
            diff = self.base_forward(xa_i, xb_i, i)
            gate = self.gate(self.gap_sigmoid(diff)) # [B, C, 1, 1]
            out = diff * gate # [B, C, H, W]
            outs.append(out)
        return outs
