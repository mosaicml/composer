# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.models import MMDetModel
from composer.metrics import MAP

__all__ = ['composer_yolox']

# yapf disable


def composer_yolox(model_name: str, num_classes=80):
    """
    Args:
        model_name: (str) one of 'yolox-s', 'yolox-m', 'yolox-l', 'yolox-x'
        num_classes: (int) Default: 80
    """
    from mmcv import ConfigDict
    from mmdet.models import build_detector

    model_names = {'yolox-s', 'yolox-m', 'yolox-l', 'yolox-x'}

    yolox_s_config = dict(
        type='YOLOX',
        input_size=(640, 640),
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
        neck=dict(type='YOLOXPAFPN', in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
        bbox_head=dict(type='YOLOXHead', num_classes=num_classes, in_channels=128, feat_channels=128),
        train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        # In order to align the source code, the threshold of the val phase is
        # 0.01, and the threshold of the test phase is 0.001.
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

    if model_name == 'yolox-s':
        config = yolox_s_config

    elif model_name == 'yolox-m':  # override some architecture settings for yolox-medium size
        yolox_m_config = dict(
            backbone=dict(deepen_factor=0.67, widen_factor=0.75),
            neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
            bbox_head=dict(in_channels=192, feat_channels=192),
        )
        config = yolox_s_config.update(yolox_m_config)

    elif model_name == 'yolox-l':
        yolox_l_config = dict(backbone=dict(deepen_factor=1.0, widen_factor=1.0),
                              neck=dict(in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
                              bbox_head=dict(in_channels=256, feat_channels=256))
        config = yolox_s_config.update(yolox_l_config)

    elif model_name == 'yolox-x':
        yolox_x_config = dict(backbone=dict(deepen_factor=1.33, widen_factor=1.25),
                              neck=dict(in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
                              bbox_head=dict(in_channels=320, feat_channels=320))
        config = yolox_s_config.update(yolox_x_config)

    else:
        raise ValueError(f'model name must be one of {model_names}.')

    config = ConfigDict(config)
    metrics = [MAP(box_format='xyxy')]
    model = build_detector(config)
    model.init_weights()

    return MMDetModel(model=model, metrics=metrics)

    # yapf: enable
