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
model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='FocalNet',
        in_chans=3,
        embed_dim=embed_dim,
        depths=[2, 2, 6, 2],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,    
        focal_windows=[9, 9, 9, 9],
        focal_levels=[3, 3, 3, 3],
        out_indices=(0, 1, 2, 3),
        patch_size=4,
    ),
    # decode_head=dict(
    #     type='mmseg.FCNHead',
    #     in_channels=embed_dim * 4,
    #     channels=embed_dim * 4,
    #     in_index=-1,
    #     num_convs=0,
    #     concat_input=False,
    #     num_classes=2,
    #     loss_decode=dict(
    #         type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        type='mmseg.UPerHead',
        # in_channels=[embed_dim * 4] * 4,
        in_channels=[v for v in [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # neck=dict(type='FeatureFusionNeck', policy='diff'),
    neck=dict(type='FocalFusion', in_channels=[embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]),
    # auxiliary_head=dict(
    #     type='mmseg.FCNHead',
    #     in_channels=embed_dim * 4,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
