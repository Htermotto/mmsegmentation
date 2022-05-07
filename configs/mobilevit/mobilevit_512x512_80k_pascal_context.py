_base_ = [
    '../_base_/models/deeplabv3_mobilevit-d8.py', '../_base_/datasets/pascal_context.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        type='MobileViT',
        num_classes=60,
    ),
    decode_head=dict(num_classes=60),
    auxiliary_head=dict(num_classes=60),
)
