# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping, Sequence
from itertools import chain

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

__all__ = ['coco_mmdet', 'mmdet_collate', 'mmdet_get_num_samples']


def coco_mmdet(path: str = '/data/coco', split: str = 'train'):
    """creates a coco dataset in mmdetection format

    Args:
        path (str): path to unzipped coco data. Default: ``''/data/coco'```.
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
    """
    from mmcv import ConfigDict
    from mmdet.datasets import build_dataset

    ann_file = path + f'/annotations/instances_{split}2017.json'
    img_prefix = path + f'/{split}2017'
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
            'type': 'LoadAnnotations',
            'with_bbox': True
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
                'keys': ['img', 'gt_bboxes', 'gt_labels']
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
            'type': 'LoadAnnotations',
            'with_bbox': True
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
                'keys': ['img', 'gt_bboxes', 'gt_labels']
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


def mmdet_collate(batch) -> list:
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """
    from mmcv.parallel.data_container import DataContainer

    samples_per_gpu = 1

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        tensor_stack = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(batch[i].data)
            return stacked  # modify to not return data container

        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1], sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(F.pad(sample.data, pad, value=sample.padding_value))
                    tensor_stack.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    tensor_stack.append(default_collate([sample.data for sample in batch[i:i + samples_per_gpu]]))
                else:
                    raise ValueError('pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                tensor_stack.append([sample.data for sample in batch[i:i + samples_per_gpu]])
        # hack to convert for mosaic
        tensor_stack = list(chain.from_iterable(tensor_stack))  # flatten
        try:
            return torch.stack(tensor_stack)
        except RuntimeError:
            return tensor_stack

    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [mmdet_collate(samples) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {key: mmdet_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return default_collate(batch)


def mmdet_get_num_samples(batch):
    return len(batch)
