# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import torch

from torchinfo import summary

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from mmseg.models.backbones.mobilenet_v2 import MobileNetV2
from mmseg.models.backbones import MobileViT

from mmseg.models.backbones.mobilevit import mobilevit_xxs
from mmseg.models.backbones.vit import VisionTransformer
from mmseg.models.necks import MultiLevelNeck
from mmseg.models.decode_heads import UPerHead, FCNHead

import cv2
import copy


def main():
    # parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--palette',
    #     default='cityscapes',
    #     help='Color palette used for segmentation map')
    # parser.add_argument(
    #     '--opacity',
    #     type=float,
    #     default=0.5,
    #     help='Opacity of painted segmentation map. In (0, 1] range.')
    # args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor('configs/mobilevit/mobilevit_512x512_80k_ade20k.py', device='cpu')

    im = cv2.imread('demo/demo.png')
    im = torch.tensor(im)
    im = im.reshape((1, *im.shape))
    im = torch.transpose(im, 3, 1)
    im = im.float()
    print(im.shape, im.dtype)

    mobile_vit = mobilevit_xxs()

    r = mobile_vit(im)
    print(r)
    print(r.shape)

    r = tuple([copy.copy(r) for _ in range(4)])

    # vit = VisionTransformer(
    #     img_size=(512, 512),
    #     patch_size=16,
    #     in_channels=3,
    #     embed_dims=768,
    #     num_layers=12,
    #     num_heads=12,
    #     mlp_ratio=4,
    #     out_indices=(2, 5, 8, 11),
    #     qkv_bias=True,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     drop_path_rate=0.0,
    #     with_cls_token=True,
    #     norm_cfg=dict(type='LN', eps=1e-6),
    #     act_cfg=dict(type='GELU'),
    #     norm_eval=False,
    #     interpolate_mode='bicubic',
    # )
    # r = vit(im)
    print('r len: {len(r)}')
    for tensor in r:
        print(tensor.shape)
    print('end len')
    # print(r[0].shape)
    neck = MultiLevelNeck(
        in_channels=[320, 320, 320, 320],
        out_channels=320,
        scales=[4, 2, 1, 0.5],
    )
    print('PASSING TUPLE!')
    r = neck(r)
    print(type(r))
    print(len(r))
    for tensor in r:
        print(tensor.shape)
    norm_cfg = dict(type='SyncBN', requires_grad=True)
    uper = UPerHead(   
        in_channels=[320, 320, 320, 320],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
    
    r = uper(r)

        

    # model = init_segmentor('configs/vit/upernet_deit-s16_512x512_80k_ade20k.py', device='cpu')
    # summary(model)

    # # test a single image
    # result = inference_segmentor(model, 'demo/demo.png')

    # # show the results
    # show_result_pyplot(
    #     model,
    #     'demo/demo.png',
    #     result,
    #     get_palette('cityscapes'),
    #     opacity=0.5)


if __name__ == '__main__':
    main()
