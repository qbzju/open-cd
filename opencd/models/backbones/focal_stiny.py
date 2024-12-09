import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from opencd.registry import MODELS
from .focal_modulation import PatchEmbed
from ..necks.focal_fusion import CrossFocalModulation, ChannelAttention

class Mlp(nn.Module):
    """Channel mixing layer with 1x1 convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        mlp_ratio (float): Channel expansion ratio. Default: 4.
        drop (float): Dropout rate. Default: 0.
        use_residual (bool): Whether to use residual connection. Default: True
    """
    def __init__(self, 
                 in_channels: int, 
                 mlp_ratio: float = 4., 
                 drop: float = 0.,
                 use_residual: bool = True):
        super().__init__()
        
        hidden_channels = int(in_channels * mlp_ratio)
        self.use_residual = use_residual
        
        self.channel_mix = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.Dropout(p=drop)
        )
    
    def forward(self, x):
        if self.use_residual:
            return self.channel_mix(x) + x
        return self.channel_mix(x)


@MODELS.register_module()
class StinyFocalNet(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 mlp_ratio=4.,
                 drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 focal_levels=[2, 2, 2, 2],
                 focal_windows=[9, 9, 9, 9],
                 use_conv_embed=False,
                 normalize_context=False,
                 num_head=4,
                ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.num_layers = len(focal_levels)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            use_conv_embed=use_conv_embed, is_stem=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # build layers
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        self.layers = nn.ModuleList()
        self.channel_attentions = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # TODO: add multi-head cross focal modulation
            layer = CrossFocalModulation(
                    dim=2 * self.num_features[i_layer],
                    focal_level=focal_levels[i_layer],
                    focal_window=focal_windows[i_layer],
                    normalize_context=normalize_context,
                    drop=dpr[i_layer],
                    num_head=num_head
                )
            ca = ChannelAttention(2 * self.num_features[i_layer])
            mlp = Mlp(
                in_channels=2 * self.num_features[i_layer],
                mlp_ratio=mlp_ratio,
                drop=dpr[i_layer],
                use_residual=True
            )
            
            self.layers.append(layer)
            self.channel_attentions.append(ca)
            self.mlps.append(mlp)

        self.downsamples = nn.ModuleList()
        for i in range(self.num_layers):
            if (i < self.num_layers - 1):
                self.downsamples.append(PatchEmbed(
                    patch_size=2,
                    in_chans=self.num_features[i], 
                    embed_dim=2 * self.num_features[i],
                    use_conv_embed=use_conv_embed,
                    norm_layer=norm_layer,
                    is_stem=False
                )) 

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

    def forward(self, xA, xB):
        """Forward function."""
        assert xA.shape == xB.shape, "xA and xB must have the same shape"
        xA = self.patch_embed(xA)
        xB = self.patch_embed(xB)
        B, C, H, W = xA.shape

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            mlp = self.mlps[i]
            # cross focal modulation
            xA, xB, fusion = layer(xA, xB)                                 
            fusion = self.channel_attentions[i](fusion) * fusion    # [B, C, H, W]
            fusion = mlp(fusion)                                    # residual connection

            if i in self.out_indices:
                outs.append(fusion)

            if (i < self.num_layers - 1):
                xA = self.downsamples[i](xA)
                xB = self.downsamples[i](xB)
                H, W = (H + 1) // 2, (W + 1) // 2


        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(StinyFocalNet, self).train(mode)
        self._freeze_stages()
