_base_ = [
    '../_base_/models/focalnet_c96.py',
    '../common/standard_256x256_40k_levircd.py']

data_root = '../datasets/LEVIR-CD'

embed_dim = 96
patch_size = 4

model = dict(
    type='DIEncoderDecoder',
    backbone=dict(
        type='StinyFocalNet',
        patch_size=patch_size,
        embed_dim=embed_dim,
        focal_windows=[3, 3, 1],
        focal_levels=[6, 6, 6],
        out_indices=(0, 1, 2),
        normalize_context=True,
        drop_path_rate=0.2,
        num_head=4,
    ),
    decode_head=dict(
        _delete_=True,
        type='StinyFocalDecoder',
        in_channels=[v * 2 for v in [embed_dim, embed_dim*2, embed_dim*4]],
        embed_dim=96,
        mlp_ratio=4.,
        drop_path_rate=0.2,
        frozen_stages=-1,
        focal_levels=[6, 6, 6],
        focal_windows=[3, 3, 1],
        normalize_context=True,
        num_head=4,
        num_classes=2,
        input_transform='resize_concat',
        in_index=(0, 1, 2),
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
        end=1000,
    ),
    # fast-decay
    dict(
        type='CosineAnnealingLR',
        T_max=4000,
        by_epoch=False,
        begin=1000,
        end=5000,
        eta_min=1e-5,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=15000,
        by_epoch=False,
        begin=5000,
        end=20000,
        eta_min=1e-6,
    ),
    # fine-tune
    dict(
        type='CosineAnnealingLR',
        T_max=20000,
        by_epoch=False,
        begin=20000,
        end=40000,
        eta_min=1e-8,
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
