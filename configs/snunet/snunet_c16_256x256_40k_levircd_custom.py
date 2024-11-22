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
                 eps=1e-8, 
                )


optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,
#                     save_best='mIoU'),
# )

# custom_hooks = [
#     dict(type='ModelCheckHook', interval=1)
# ]