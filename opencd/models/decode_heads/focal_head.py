# ------------------------------------------------------------------
# 2D Focal Modulation Network Decoder
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>

# The original code is available at https://github.com/microsoft/FocalNet/tree/main
# and is licensed under The MIT License. Copyright (c) 2022 Microsoft
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from opencd.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

# ------------------------------------------------------------


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1,
                           stride=1, padding=0, bias=False)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate_sm = nn.Softmax(dim=1)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection

        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # use softmax to normalize the gates
        gates = self.gate_sm(gates)

        # context aggreation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]

        # change ctx to ctx_all
        ctx_global = self.act(ctx_all.mean(
            2, keepdim=True).mean(3, keepdim=True))
        ctx_all = (ctx_all + ctx_global *
                   gates[:, self.focal_level:]) / (self.focal_level + 1)

        # normalize context
        # if self.normalize_modulator:
        #     ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)

        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()

        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)

        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        flops += N * self.dim * (self.dim * 2 + (self.focal_level + 1))

        # focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k] ** 2 + 1) * self.dim

        # global gating
        flops += N * 1 * self.dim

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class FocalNetBlock(nn.Module):
    """ Focal Modulation Network Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether to use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether to use layernorm after modulation
    """

    def __init__(self, dim, out_dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU,
                 focal_level=1, focal_window=3,
                 use_layerscale=False, layerscale_value=1e-4,
                 use_postln=False, use_postln_in_modulation=False,
                 normalize_modulator=False,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = nn.LayerNorm(dim)

        self.modulation = FocalModulation(
            dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level,
            use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=out_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(
                layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                layerscale_value * torch.ones((dim)), requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape

        # remove shortcut

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H * W, C)
        x = x if not self.use_postln else self.norm1(x)

        # FFN
        x = self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(
            self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))

        return x


class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        upsample (nn.Module | None, optional): Upsampling layer at the end of the layer. Default: None
        upsampling_type (str): Upsampling type. Default: bilinear
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether to use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether to use layernorm after modulation
    """

    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 upsample=None, upsampling_type='bilinear', use_checkpoint=False,
                 focal_level=1, focal_window=1,
                 use_layerscale=False, layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.upsampling_type = upsampling_type

        # TODO: add linear gate
        self.f = nn.Linear(dim, 2*dim+(self.depth+1), bias=True)
        self.act = nn.GELU()
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1,
                           padding=0, groups=1, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.gate_sm = nn.Softmax(dim=2)

        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=dim,
                out_dim=None,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
            )
            for i in range(depth)])

        if upsample is not None:
            self.upsample = upsample(
                upsampling=upsampling_type,
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
            )
        else:
            self.upsample = None

    def forward(self, x, H, W):

        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        Ho, Wo = H, W

        B, L, C = x.shape

        x = self.f(x)
        # B L, (C, C, self.depth+1)
        q, ctx, gates = torch.split(x, (C, C, self.depth+1), 2)
        # use softmax to normalize the gates
        gates = self.gate_sm(gates)

        ctx_all = 0

        for i, blk in enumerate(self.blocks):
            blk.H, blk.W = Ho, Wo
            ctx = blk(ctx)
            ctx_all = ctx_all + ctx * gates[:, :, i:i+1]

        ctx_global = self.act(ctx_all.mean(dim=1, keepdim=True))
        ctx_all = (ctx_all + ctx_global *
                   gates[:, :, self.depth:]) / (self.depth + 1)

        q = q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_out = q * self.h(ctx_all.view(B, H, W,
                           C).permute(0, 3, 1, 2).contiguous())

        x_out = x_out.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
        x_out = self.proj(x_out)  # [B, L, C]
        x_out = self.proj_drop(x_out)
        x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.upsample is not None:
            x_out, Ho, Wo = self.upsample(x_out)
        else:
            Ho, Wo = H, W

        return x_out, Ho, Wo


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        upsampling (str): type of the up sampling conv or bilinear.  Default: bilinear.
        patch_size (int): Patch token size. Default: 2.
        in_chans (int): Number of input image channels. Default: 96.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, upsampling='bilinear', patch_size=2, in_chans=96, embed_dim=96):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.up_sampling = upsampling

        if self.up_sampling == 'conv':
            self.proj = nn.Sequential(nn.ConvTranspose2d(self.in_chans, self.embed_dim,
                                                         kernel_size=(
                                                             patch_size, patch_size),
                                                         stride=(patch_size, patch_size), padding=(0, 0), bias=True))
            # nn.ReLU(inplace=True))

        elif self.up_sampling in ('bilinear', 'nearest', 'bicubic'):

            self.proj = nn.Sequential(nn.Upsample(scale_factor=patch_size, mode=self.up_sampling),
                                      nn.Conv2d(self.in_chans,  self.embed_dim, kernel_size=(1, 1),
                                                stride=(1, 1), padding=(0, 0), bias=True))
            # nn.ReLU(inplace=True))
        else:
            raise ValueError('%s is not a recognized down_sampling type.\
             Supported are conv, bilinear, nearest or bicubic' % self.up_sampling)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        return x, H, W


@MODELS.register_module()
class FocalNetDecoder(BaseDecodeHead):
    """ 2D Focal Modulation Networks (FocalNets) Decoder """

    def __init__(self,
                 pretrain_img_size=1024,
                 patch_size=4,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 focal_levels=[2, 2, 2, 2],
                 focal_windows=[9, 9, 9, 9],
                 use_layerscale=False,
                 use_checkpoint=False,
                 layerscale_value=1/torch.sqrt(torch.tensor(2)),
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=True,
                 up_sampling='bilinear',
                 encoder_channels=[96, 192, 384, 768],
                 norm_layer=nn.LayerNorm,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        config : argparse
          configuration file from config.py
        """

        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint

        self.patch_size = patch_size

        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.use_layerscale = use_layerscale
        self.layerscale_value = layerscale_value
        self.use_postln = use_postln
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.up_sampling = up_sampling

        self.n_layers = len(encoder_channels)

        self.decoder_out_channels = []
        self.in_channels = [encoder_channels[-1]]

        for i in reversed(range(self.n_layers - 1)):
            if i == (self.n_layers - 2):
                self.decoder_out_channels.append(encoder_channels[i+1]//2)
            else:
                self.decoder_out_channels.append(encoder_channels[i+1])
            self.in_channels.append(
                encoder_channels[i] + self.decoder_out_channels[(self.n_layers - 2) - i])
        self.decoder_out_channels.append(encoder_channels[0]*3)

        super().__init__(in_channels=self.in_channels,
                         channels=self.decoder_out_channels[-1],
                         **kwargs)

        patches_resolution = [pretrain_img_size //
                              self.patch_size, pretrain_img_size // self.patch_size]
        self.patches_resolution = patches_resolution

        self.input_resolution = []
        for i in reversed(range(self.n_layers)):
            self.input_resolution.append(
                (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)))

        # stochastic depth
        dpr = [x.item() for x in reversed(torch.linspace(
            # stochastic depth decay rule
            0, self.drop_path_rate, sum(self.depths)))]

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.n_layers):
            layer = BasicLayer(dim=self.in_channels[i_layer],
                               out_dim=self.decoder_out_channels[i_layer],
                               input_resolution=self.input_resolution[i_layer],
                               depth=self.depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=self.drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(
                                   self.depths[:i_layer + 1])],
                               upsample=PatchEmbed if (
                                   i_layer != (self.n_layers - 1)) else None,
                               upsampling_type=self.up_sampling,
                               focal_level=self.focal_levels[i_layer],
                               focal_window=self.focal_windows[i_layer],
                               use_checkpoint=self.use_checkpoint,
                               use_layerscale=self.use_layerscale,
                               layerscale_value=self.layerscale_value,
                               use_postln=self.use_postln,
                               use_postln_in_modulation=self.use_postln_in_modulation,
                               normalize_modulator=self.normalize_modulator,
                               )

            self.layers.append(layer)

        for i_layer in range(self.n_layers):
            layer = norm_layer(self.decoder_out_channels[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_func)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {''}

    def _forward_feature(self, inputs):
        """
        Args:
            x : list of input torch.tensor [N, C, H, W]
        """

        _, _, H, W = inputs[-1].shape

        for i, x_i in enumerate(reversed(inputs)):
            if i != 0:

                if x_i.size()[-2:] != x_out.size()[-2:]:
                    x_out = F.interpolate(x_out, size=x_i.size(
                    )[-2:], mode='bilinear', align_corners=False)

                x_i = torch.cat((x_i, x_out), dim=1)

            _, _, H, W = x_i.shape

            x_out, H, W = self.layers[i](x_i, H, W)
            # normalize the output
            B, C, H, W = x_out.shape
            x_out = x_out.permute(0, 2, 3, 1).contiguous()
            x_out = getattr(self, f'norm{i}')(x_out)
            x_out = x_out.permute(0, 3, 1, 2).contiguous()

        if self.patch_size != 1:
            x_out = F.interpolate(
                x_out, scale_factor=self.patch_size, mode='bilinear', align_corners=False)

        return x_out

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)

        return output
