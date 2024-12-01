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
        focal_windows=[3, 3],
        focal_levels=[2, 2],
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
    batch_size=32,
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
optimizer = dict(_delete_=True, type='AdamW', lr=0.00003, betas=(0.9, 0.999), weight_decay=0.01,
                 eps=1e-8,
                 )


optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
