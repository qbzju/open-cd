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