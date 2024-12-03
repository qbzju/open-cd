# --------------------------------------------------------
# FocalNet for Semantic Segmentation
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang
# --------------------------------------------------------
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from opencd.registry import MODELS

class FocalKernel(nn.Module):
    """Focal Kernel for extracting features at different scales
    
    Args:
        dim (int): Number of input channels
        kernel_size (int): Size of the convolution kernel
        use_gelu (bool, default=True): Whether to use GELU activation
    """
    def __init__(self, dim, kernel_size, use_gelu=True):
        super().__init__()
        self.conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            groups=dim,  
            padding=kernel_size//2,
            bias=False
        )
        
        self.act = nn.GELU() if use_gelu else nn.Identity()
        self.H = None
        self.W = None
        
    def forward(self, x):
        '''x : [B, L, C]'''
        B, L, C = x.shape
        assert L == self.H * self.W, "input feature has wrong size"
        x_out = x.view(x.shape[0], self.H, self.W, x.shape[2]).permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        x_out = self.act(self.conv(x_out))
        return x_out.view(B, C, L).permute(0, 2, 1).contiguous() # [B, L, C]

class Aggregator(nn.Module):
    """Aggregate context features at multiple focal levels and global context
    
    Args:
        dim (int): Number of input channels
        depth (int): Number of aggregation levels
        kernel (nn.Module): Kernel module
        kwargs: Additional keyword arguments
    """
    def __init__(self, 
                 dim, 
                 depth, 
                 kernel=FocalKernel,
                 normalize_context=False,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.kernel = kernel
        self.normalize_context = normalize_context
        window = kwargs.get('focal_window', 7)
        # multi-scale convolution layers
        self.layers = nn.ModuleList()
        if kernel is FocalModulationBlock:
            for k in range(depth):
                drop_path = kwargs.get('drop_path', 0.0)
                kwargs['drop_path'] = drop_path[k] if isinstance(drop_path, list) else drop_path
                self.layers.append(kernel(dim, **kwargs))
        elif kernel is FocalKernel:
            focal_factor = kwargs.get('focal_factor', 2)
            for k in range(depth):
                kernel_size = focal_factor * k + window
                self.layers.append(kernel(dim, kernel_size))
        else: # cross-focal
            focal_factor = kwargs.get('focal_factor', 2)
            for k in range(depth):
                kernel_size = focal_factor * k + window
                self.layers.append(kernel(dim, kernel_size))

        self.act = nn.GELU()
        self.h = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=True)
        
    def forward(self, context, gates, H=None, W=None):
        """
        Args:
            context: context features  [B, L, C]
            gates: modulation gates  [B, L, focal_level+1]
        Returns:
            aggregated context features  [B, L, C]
        """
        B, L, C = context.shape
        assert L == H * W, "input feature has wrong size"
        ctx_all = 0
        ctx = context
        for i, layer in enumerate(self.layers):
            layer.H, layer.W = H, W
            ctx = layer(ctx)
            ctx_all = ctx_all + ctx * gates[:, :, i:i+1]

        ctx_global = self.act(ctx_all.mean(dim=1, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, :, self.depth:]

        if self.normalize_context:
            ctx_all = ctx_all / (self.depth + 1)

        ctx_all = ctx_all.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        ctx_all = self.h(ctx_all)
        return ctx_all.view(B, -1, L).permute(0, 2, 1).contiguous() # [B, L, C]
        
    

class Modulator(nn.Module):
    """Modulator for generating modulated features

    Args:
        dim (int): Number of input channels
        focal_level (int): Number of focal levels
    """

    def __init__(self, dim, focal_level):
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level

        self.f = nn.Linear(dim, 2*dim+(focal_level+1))
        # TODO: replace softmax with other activation function
        self.gate_sm = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Args:
            x: input features with shape (B, L, C)
        Returns:
            [B, L, X]
            query: query features
            context: context features
            gates: modulation gates for different focal levels
        """
        x = self.f(x)  # [B, L, 2C+(focal_level+1)]

        query, context, gates = torch.split(
            x,
            (self.dim, self.dim, self.focal_level+1),
            dim=2
        )

        # use softmax to normalize the gates
        gates = self.gate_sm(gates)

        return query, context, gates


class Mlp(nn.Module):
    """ Multilayer perceptron."""

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
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=True): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, 
                 focal_window=7, focal_factor=2,
                 use_postln=False, normalize_context=True):

        super().__init__()

        self.dim = dim
        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.modulator = Modulator(dim, focal_level)
        self.aggregator = Aggregator(dim, 
                                     focal_level, 
                                     normalize_context=normalize_context,
                                     focal_window=focal_window, 
                                     focal_factor=focal_factor)
        if self.use_postln:
            self.norm = nn.LayerNorm(dim)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, H=None, W=None):
        """ Forward function.

        Args:
            x: input features with shape of (B, L, C)
            x_out: output features with shape of (B, L, C)
        """
        q, ctx, gates = self.modulator(x) 
        ctx_all = self.aggregator(ctx, gates, H, W)
        x_out = q * ctx_all

        if self.use_postln:
            x_out = self.norm(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=9, use_layerscale=False, layerscale_value=1e-4, normalize_context=False,
                 use_postln=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_layerscale = use_layerscale
        self.use_postln = use_postln

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim, focal_window=self.focal_window, focal_level=self.focal_level, proj_drop=drop, normalize_context=normalize_context,
            use_postln=use_postln
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(
                (self.gamma_1 + layerscale_value) * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                (self.gamma_2 + layerscale_value) * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, L, C).
        """
        # remove shortcut
        x = x if self.use_postln else self.norm1(x)

        # FM
        x = self.modulation(x, self.H, self.W)

        # FFN
        x = self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic focal modulation layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch embedding or now. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 focal_window=9,
                 focal_level=2,
                 use_conv_embed=False,
                 use_layerscale=False,
                 use_checkpoint=False,
                 use_postln=False, # use postln in BaseLayer
                 normalize_context=False,
                ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_postln = use_postln
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        # self.normalize_layer = normalize_layers
        self.modulator = Modulator(dim, depth)
        self.aggregator = Aggregator(dim, depth, 
                                     FocalModulationBlock,
                                     drop=drop,
                                     drop_path=drop_path,
                                     focal_window=focal_window,
                                     focal_level=focal_level,
                                     use_layerscale=use_layerscale,
                                     norm_layer=norm_layer,
                                     use_postln=use_postln,
                                     normalize_context=normalize_context,
                                     mlp_ratio=mlp_ratio)
        # TODO : stochastic depth decay rule
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                patch_size=2,
                in_chans=dim, embed_dim=2*dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False
            )

        else:
            self.downsample = None
        
        if self.use_postln:
            self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        q, ctx, gates = self.modulator(x)
        ctx_all = self.aggregator(ctx, gates, H, W)

        x_out = q * ctx_all

        if self.use_postln:
            x_out = self.norm(x_out)

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if self.downsample is not None:
            x_reshaped = x_out.transpose(1, 2).view(
                x_out.shape[0], x_out.shape[-1], H, W)
            x_down = self.downsample(x_reshaped)
            x_down = x_down.flatten(2).transpose(1, 2)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_out, H, W, x_down, Wh, Ww
        else:
            return x_out, H, W, x_out, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not. 
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, 
                 norm_layer=None, use_conv_embed=False, is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7
                padding = 3
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(
                x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


@MODELS.register_module()
class FocalNet(nn.Module):
    """ FocalNet backbone.

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each FocalNet stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level at four stages
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=1024,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 focal_levels=[2, 2, 2, 2],
                 focal_windows=[9, 9, 9, 9],
                 use_conv_embed=False,
                 use_layerscale=False,
                 use_postln=False,
                 normalize_context=False,
                 use_checkpoint=False,
                 use_downsample=True,
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            use_conv_embed=use_conv_embed, is_stem=True)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                # stochastic depth decay rule
                                                sum(depths))]

        # TODO: add linear gate

        # build layers
        self.layers = nn.ModuleList()
        if use_downsample:
            for i_layer in range(self.num_layers):
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchEmbed if (
                        i_layer < self.num_layers - 1) else None,
                    focal_window=focal_windows[i_layer],
                    focal_level=focal_levels[i_layer],
                    use_conv_embed=use_conv_embed,
                    use_layerscale=use_layerscale,
                    use_checkpoint=use_checkpoint,
                    use_postln=use_postln,
                    normalize_context=normalize_context,
                )
                self.layers.append(layer)
        else:
            for i_layer in range(self.num_layers):
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    focal_window=focal_windows[i_layer],
                    focal_level=focal_levels[i_layer],
                    use_conv_embed=use_conv_embed,
                    use_layerscale=use_layerscale,
                    use_checkpoint=use_checkpoint,
                    use_postln=use_postln,
                    normalize_context=normalize_context,
                )
                self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i)
                        for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W,
                                 self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(FocalNet, self).train(mode)
        self._freeze_stages()
