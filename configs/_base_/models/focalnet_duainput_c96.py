# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))
embed_dim = 96
patch_size = 4
model = dict(
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='FocalNet',
        in_chans=3,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depths=[1, 1],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
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
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    neck=dict(type='FocalFusion',
              in_channels=[v * 2 for v in [embed_dim, embed_dim*2]],
              focal_factor=2,
              focal_window=[3, 3],
              focal_level=[2, 2],
              drop=0.0),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
