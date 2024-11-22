_base_ = [
    '../_base_/models/snunet_c16_custom.py',
    '../common/standard_256x256_40k_levircd.py']

data_root = '../datasets/LEVIR-CD'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root  
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root 
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 eps=1e-8, clip_norm=1.0,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

# custom_hooks = [
#     dict(type='ModelCheckHook', interval=1)
# ]