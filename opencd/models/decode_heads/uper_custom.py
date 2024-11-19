from mmseg.models.decode_heads import UPerHead
from opencd.registry import MODELS
import torch.nn as nn

@MODELS.register_module()
class CustomUPerHead(UPerHead):
    def __init__(self, patch_size=4, **kwargs):
        super().__init__(**kwargs)
        self.patch_to_pixel = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels,  
                self.channels // patch_size,
                kernel_size=patch_size, 
                stride=patch_size
            ),
            nn.BatchNorm2d(self.channels // patch_size),
            nn.ReLU(inplace=True)
        )
        
        self.conv_seg = nn.Conv2d(
            self.channels // patch_size,  
            self.num_classes,
            kernel_size=1
        )

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.patch_to_pixel(output)
        output = self.cls_seg(output)

        return output