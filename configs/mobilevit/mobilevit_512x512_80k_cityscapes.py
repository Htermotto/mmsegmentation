_base_ = [
    '../_base_/models/upernet_mobilevit-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='',
    backbone=dict(
        type='MobileViT',
    ),
    decode_head=dict(num_classes=19),
    auxiliary_head=dict(num_classes=19),
)
