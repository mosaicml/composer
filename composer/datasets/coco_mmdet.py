# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0


def coco_mmdet(ann_file: str, img_prefix: str, split: str):
    config = dict()
    # config.data.train.dataset.ann_file = '../mmdetection/data/coco/annotations/instances_train2017.json'
    # config.data.train.dataset.img_prefix = '../mmdetection/data/coco/train2017'
    config.data.train.dataset.ann_file = ann_file
    config.data.train.dataset.img_prefix = img_prefix

    from mmdet.datasets import build_dataset
    if split == 'train':
        return build_dataset(config.data.train)
    if split == 'val':
        return build_dataset(config.data.val)
    if split == 'test':
        return build_dataset(config.data.test)
