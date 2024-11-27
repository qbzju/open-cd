import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from opencd.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from ..backbones.focal_modulation import FocalNet

    

@MODELS.register_module()
class FocalNetDecoder(FocalNet, BaseDecodeHead):
    """ 2D Focal Modulation Networks (FocalNets) Decoder """

    def __init__(self, 
                 embed_dim=96,
                 up_sampling='bilinear',
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 patch_size=4,
                 depths=[2, 2, 6, 2],
                 focal_levels=[2, 2, 2, 2],
                 focal_windows=[9, 9, 9, 9],
                 encoder_channels=[96, 192, 384, 768],
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        
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


        BaseDecodeHead.__init__(self, in_channels=self.in_channels,
                         channels=self.decoder_out_channels[-1],
                         **kwargs)
        FocalNet.__init__(self,
                         patch_size=patch_size,
                         embed_dim=embed_dim,
                         depths=depths,
                         mlp_ratio=mlp_ratio,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer,
                         focal_levels=focal_levels,
                         focal_windows=focal_windows,
                         use_downsample=False,
                        )
        dpr = [x.item() for x in reversed(torch.linspace(
            # stochastic depth decay rule
            0, self.drop_path_rate, sum(self.depths)))]

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


    def _forward_feature(self, inputs):
        """
        Args:
            x : list of input torch.tensor [N, C, H, W]
        """

        for i, x_i in enumerate(reversed(inputs)):
            if i != 0:

                if x_i.size()[-2:] != x_out.size()[-2:]:
                    x_out = F.interpolate(x_out, size=x_i.size(
                    )[-2:], mode='bilinear', align_corners=False)

                x_i = torch.cat((x_i, x_out), dim=1)

            _, _, H, W = x_i.shape

            x_out, _, _, _, _, _ = self.layers[i](x_i, H, W)
            # normalize the output
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
