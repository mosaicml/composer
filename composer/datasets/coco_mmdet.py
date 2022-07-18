# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

__all__ = ['coco_mmdet']


def coco_mmdet(path: str = '/data/coco', split: str = 'train'):
    from mmcv import ConfigDict
    from mmdet.datasets import build_dataset

    ann_file = path + f'/annotations/instances_{split}2017.json'
    img_prefix = path + '/train2017'
    train_config = {
        'type':
            'MultiImageMixDataset',
        'dataset': {
            'type': 'CocoDataset',
            'ann_file': ann_file,
            'img_prefix': img_prefix,
            'pipeline': [{
                'type': 'LoadImageFromFile'
            }, {
                'type': 'LoadAnnotations',
                'with_bbox': True
            }],
            'filter_empty_gt': False
        },
        'pipeline': [{
            'type': 'Mosaic',
            'img_scale': (640, 640),
            'pad_val': 114.0
        }, {
            'type': 'RandomAffine',
            'scaling_ratio_range': (0.1, 2),
            'border': (-320, -320)
        }, {
            'type': 'MixUp',
            'img_scale': (640, 640),
            'ratio_range': (0.8, 1.6),
            'pad_val': 114.0
        }, {
            'type': 'YOLOXHSVRandomAug'
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.5
        }, {
            'type': 'Resize',
            'img_scale': (640, 640),
            'keep_ratio': True
        }, {
            'type': 'Pad',
            'pad_to_square': True,
            'pad_val': {
                'img': (114.0, 114.0, 114.0)
            }
        }, {
            'type': 'FilterAnnotations',
            'min_gt_bbox_wh': (1, 1),
            'keep_empty': False
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img', 'gt_bboxes', 'gt_labels']
        }]
    }

    val_config = {
        'type':
            'CocoDataset',
        'ann_file':
            ann_file,
        'img_prefix':
            img_prefix,
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type':
                'MultiScaleFlipAug',
            'img_scale': (640, 640),
            'flip':
                False,
            'transforms': [{
                'type': 'Resize',
                'keep_ratio': True
            }, {
                'type': 'RandomFlip'
            }, {
                'type': 'Pad',
                'pad_to_square': True,
                'pad_val': {
                    'img': (114.0, 114.0, 114.0)
                }
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }]
        }]
    }

    test_config = {
        'type':
            'CocoDataset',
        'ann_file':
            ann_file,
        'img_prefix':
            img_prefix,
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type':
                'MultiScaleFlipAug',
            'img_scale': (640, 640),
            'flip':
                False,
            'transforms': [{
                'type': 'Resize',
                'keep_ratio': True
            }, {
                'type': 'RandomFlip'
            }, {
                'type': 'Pad',
                'pad_to_square': True,
                'pad_val': {
                    'img': (114.0, 114.0, 114.0)
                }
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }]
        }]
    }

    from mmdet.datasets import build_dataset
    if split == 'train':
        return build_dataset(ConfigDict(train_config))
    if split == 'val':
        return build_dataset(ConfigDict(val_config))
    if split == 'test':
        return build_dataset(ConfigDict(test_config))
