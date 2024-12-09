import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from opencd.registry import MODELS
from ..necks.focal_fusion import CrossFocalModulation, ChannelAttention
from ..backbones.focal_stiny import Mlp
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


@MODELS.register_module()
class StinyFocalDecoder(BaseDecodeHead):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 embed_dim=96,
                 mlp_ratio=4.,
                 drop_path_rate=0.5,
                 frozen_stages=-1,
                 focal_levels=[2, 2, 2, 2],
                 focal_windows=[9, 9, 9, 9],
                 normalize_context=False,
                 num_head=4,
                 **kwargs
                ):
        super().__init__(in_channels, 2 * embed_dim,**kwargs)

        self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.num_layers = len(focal_levels)

        assert len(in_channels) == self.num_layers and len(focal_windows) == self.num_layers, \
            "in_channels and focal_windows must have the same length"

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers - 1)]

        # build layers
        self.upsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.channel_attentions = nn.ModuleList()
        self.mlps = nn.ModuleList()
        in_channels = in_channels[::-1]
        for i_layer in range(self.num_layers - 1):
            curr_chans = in_channels[i_layer]
            next_chans = in_channels[i_layer + 1]

            upsample = Upsample(curr_chans, next_chans)
            layer = CrossFocalModulation(
                dim=2 * next_chans, 
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                normalize_context=normalize_context,
                num_head=num_head,
                fuse_mode='add',
                drop=dpr[i_layer],
            )
            ca = ChannelAttention(next_chans)
            mlp = Mlp(
                in_channels=next_chans,
                mlp_ratio=mlp_ratio,
                drop=dpr[i_layer],
                use_residual=True
            )
            
            self.upsamples.append(upsample)
            self.layers.append(layer)
            self.channel_attentions.append(ca)
            self.mlps.append(mlp)

        self.out_conv = Mlp(
            in_channels=in_channels[-1],
            mlp_ratio=mlp_ratio,
            drop=dpr[-1],
            use_residual=True
        )
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

    def _forward_feature(self, features):
        features = features[::-1] 
        x = features[0] 
        for i in range(self.num_layers - 1):
            x_up = self.upsamples[i](x)
            skip = features[i + 1]
            _, _, x = self.layers[i](x_up, skip)
            x = self.channel_attentions[i](x) * x
            x = self.mlps[i](x)
            x += skip

        out = self.out_conv(x)
        
        return out
    
    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(StinyFocalDecoder, self).train(mode)
        self._freeze_stages()
