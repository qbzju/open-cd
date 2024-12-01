import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from opencd.registry import MODELS
from einops.layers.torch import Rearrange
from .focal_modulation import FocalNet, PatchEmbed, FocalModulationBlock

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

        # TODO: add linear gate

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = FocalModulationBlock(
                dim=int(embed_dim * 2 ** i_layer),
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i_layer],
                norm_layer=norm_layer,
                focal_window=focal_windows[i_layer],
                focal_level=focal_levels[i_layer],
                use_layerscale=use_layerscale,
                use_postln=use_postln,
                normalize_context=normalize_context,
            )
            self.layers.append(layer)


        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.downsamples = []
        for i in range(self.num_layers):
            if (i < self.num_layers - 1):
                self.downsamples.append(PatchEmbed(
                    patch_size=2,
                    in_chans=num_features[i], 
                    embed_dim=2 * num_features[i],
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
        B, C, H, W = xA.shape
        xA = self.patch_embed(xA)
        xB = self.patch_embed(xB)

        xA = xA.flatten(2).transpose(1, 2)  # [B, H*W, C]
        xB = xB.flatten(2).transpose(1, 2)  # [B, H*W, C]
        xA = self.pos_drop(xA)
        xB = self.pos_drop(xB)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer.H, layer.W = H, W
            x_outA = layer(xA)
            x_outB = layer(xB)
            # TODO: focal fusion
            diff = torch.abs(x_outA - x_outB)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                diff = norm_layer(diff)

                out = diff.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
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
