_base_ = [
    '../_base_/models/focalnet_c96.py',
    '../common/standard_256x256_40k_levircd.py']

data_root = '../datasets/LEVIR-CD'

embed_dim = 192
patch_size = 4

model = dict(
    type='DIEncoderDecoder',
    backbone=dict(
        type='StinyFocalNet',
        patch_size=patch_size,
        embed_dim=embed_dim,
        focal_windows=[9, 9],
        focal_levels=[6, 6],
        out_indices=(0, 1),
        normalize_context=True,
        use_postln=True,
        use_layerscale=True
    ),
    decode_head=dict(
        type='mmseg.UPerHead',
        in_channels=[v * 1 for v in [embed_dim, embed_dim*2]],
        in_index=[0, 1],
        pool_scales=(1, 2),
        channels=64,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
    ),
)


train_dataloader = dict(
    batch_size=8,
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
optimizer = dict(_delete_=True, 
                 type='AdamW', lr=0.0001, 
                 betas=(0.9, 0.999), 
                 weight_decay=0.05,
                )

param_scheduler = [
    # warm-up
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=2000,  
    ),
    # fast-decay
    dict(
        type='CosineAnnealingLR',
        T_max=18000,  
        by_epoch=False,
        begin=2000,
        end=20000,
    ),
    # fine-tune
    dict(
        type='CosineAnnealingLR',
        T_max=20000,  
        by_epoch=False,
        begin=20000,
        end=40000,
        eta_min=1e-7,  
    )
]


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=5.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
)

runner = dict(
    param_scheduler=param_scheduler 
)