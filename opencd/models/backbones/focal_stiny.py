import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from opencd.registry import MODELS
from .focal_modulation import PatchEmbed, Mlp
from ..necks.focal_fusion import CrossFocalModulation, ChannelAttention

@MODELS.register_module()
class StinyFocalNet(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
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

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                # stochastic depth decay rule
                                                self.num_layers)]

        # build layers
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        self.layers = nn.ModuleList()
        self.channel_attentions = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # TODO: add drop_path implementation, mlp, norm_layer
            layer = CrossFocalModulation(
                    dim=2 * self.num_features[i_layer],
                    focal_level=focal_levels[i_layer],
                    focal_window=focal_windows[i_layer],
                    normalize_context=normalize_context,
                    drop=dpr[i_layer]
                )
            ca = ChannelAttention(self.num_features[i_layer])
            mlp = Mlp(in_features=self.num_features[i_layer],
                    hidden_features=int(mlp_ratio * self.num_features[i_layer]),
                    act_layer=nn.GELU,
                    drop=drop_rate)
            norm = norm_layer(self.num_features[i_layer])
            
            self.layers.append(layer)
            self.channel_attentions.append(ca)
            self.mlps.append(mlp)
            self.layer_norms.append(norm)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

        xA = xA.flatten(2).transpose(1, 2)  # [B, H*W, C]
        xB = xB.flatten(2).transpose(1, 2)  # [B, H*W, C]
        xA = self.pos_drop(xA)
        xB = self.pos_drop(xB)
        xA = xA.transpose(1, 2).view(B, C, H, W).contiguous() # [B, C, H, W]
        xB = xB.transpose(1, 2).view(B, C, H, W).contiguous() # [B, C, H, W]

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            mlp = self.mlps[i]
            norm = self.layer_norms[i]
            layer.H, layer.W = H, W
            # cross focal modulation
            fusion = layer(xA, xB).flatten(2).transpose(1, 2)       # [B, L, C]
            fusion = norm(fusion).transpose(1, 2).view(B, -1, H, W).contiguous()  # [B, C, H, W]
            fusion = self.channel_attentions[i](fusion) * fusion    # [B, C, H, W]
            fusion = fusion.flatten(2).transpose(1, 2)              # [B, L, C]
            fusion = mlp(fusion) + fusion                           # residual connection

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                fusion = norm_layer(fusion)

                out = fusion.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
                outs.append(out)

            if (i < self.num_layers - 1):
                xA = self.downsamples[i](xA)
                xB = self.downsamples[i](xB)
                H, W = (H + 1) // 2, (W + 1) // 2


        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(StinyFocalNet, self).train(mode)
        self._freeze_stages()
