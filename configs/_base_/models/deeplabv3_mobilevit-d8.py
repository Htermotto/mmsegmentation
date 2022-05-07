# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MobileViT',
        image_size=((1024,512)),
        dims=[64, 80, 96],
        channels=[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        num_classes=19,
        pretrained_path='/content/mmsegmentation/pretrained_weights/deeplabv3_mobilevit_xxs.pt',
    ),
    # neck=dict(
    #     type='MultiLevelNeck',
    #     in_channels=[320, 320, 320, 320],
    #     out_channels=320,
    #     scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='ASPPHead',
        in_channels=320,
        in_index=3,
        channels=256,
        dilations=(6,12,18),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))