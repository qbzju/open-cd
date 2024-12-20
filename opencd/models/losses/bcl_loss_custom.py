# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn

from opencd.registry import MODELS

@MODELS.register_module()
class BCLLossCustom(nn.Module):
    """Batch-balanced Contrastive Loss"""

    def __init__(
            self, 
            margin=2.0, 
            loss_weight=1.0, 
            ignore_index=255,
            loss_name='bcl_loss',
            **kwargs):
        super().__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.eps = 1e-4

    def forward(self,
                pred,
                target,
                **kwargs):
        mask = (target != self.ignore_index)
        valid_target = target[mask]
        
        num_pos = (valid_target == 1).sum().float() + self.eps
        num_neg = (valid_target == 0).sum().float() + self.eps

        class_weights = torch.tensor([
            num_pos / (num_pos + num_neg),  
            num_neg / (num_pos + num_neg)
        ], device=pred.device, dtype=torch.float32)
        
        assert torch.abs(class_weights.sum() - 1.0) < 1e-6, \
            f'class weights sum should be 1, got {class_weights.sum()}'

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        losses = criterion(pred, target)
        return self.loss_weight * losses

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name