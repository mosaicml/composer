# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""COCO (Common Objects in Context) dataset.

COCO is a large-scale object detection, segmentation, and captioning dataset. Please refer to the `COCO dataset
<https://cocodataset.org>`_ for more details.
"""
import json
import os
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from composer.core import Batch, DataSpec
from composer.models.ssd.utils import DefaultBoxes, SSDTransformer
from composer.utils import MissingConditionalImportError, dist

__all__ = ['COCODetection', 'build_coco_detection_dataloader', 'build_streaming_coco_dataloader']


def build_coco_detection_dataloader(
    global_batch_size: int,
    datadir: str,
    *,
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    input_size: int = 300,
    **dataloader_kwargs: Dict[str, Any],
):
    """Builds a COCO Detection dataloader with default transforms for SSD300.

    Args:
        global_batch_size (int): Global batch size.
        datadir (str): Path to the data directory
        split (str): the dataset split to use either 'train', 'val', or 'test'. Default: ``'train```.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        input_size (int): the size of the input image, keep this at `300` for SSD300. Default: ``300``.
        **dataloader_kwargs (Any): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    # default boxes set for SSD300
    default_boxes = DefaultBoxes(fig_size=input_size,
                                 feat_size=[38, 19, 10, 5, 3, 1],
                                 steps=[8, 16, 32, 64, 100, 300],
                                 scales=[21, 45, 99, 153, 207, 261, 315],
                                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                 scale_xy=0.1,
                                 scale_wh=0.2)

    if split == 'train':
        transforms = SSDTransformer(default_boxes, (input_size, input_size), val=False, num_cropping_iterations=1)
        annotate_file = os.path.join(datadir, 'annotations/instances_train2017.json')
        image_folder = os.path.join(datadir, 'train2017')

    else:
        transforms = SSDTransformer(default_boxes, (input_size, input_size), val=True)
        annotate_file = os.path.join(datadir, 'annotations/instances_val2017.json')
        image_folder = os.path.join(datadir, 'val2017')

    dataset = COCODetection(img_folder=image_folder, annotate_file=annotate_file, transforms=transforms)

    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataSpec(dataloader=DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          sampler=sampler,
                                          drop_last=drop_last,
                                          **dataloader_kwargs),
                    split_batch=split_coco_batch)


class COCODetection(Dataset):
    """PyTorch Dataset for the COCO dataset.

    Args:
        img_folder (str): the path to the COCO folder.
        annotate_file (str): path to a file that contains image id, annotations (e.g., bounding boxes and object
            classes) etc.
        transforms (torch.nn.Module): transformations to apply to the image.
    """

    def __init__(self, img_folder: str, annotate_file: str, transforms: Optional[Callable] = None):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        # Start processing annotation
        with open(annotate_file) as fin:
            self.data = json.load(fin)

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        # 0 stands for the background
        cnt = 0
        self.label_info[cnt] = 'background'
        for cat in self.data['categories']:
            cnt += 1
            self.label_map[cat['id']] = cnt
            self.label_info[cnt] = cat['name']

        # build inference for images
        for img in self.data['images']:
            img_id = img['id']
            img_name = img['file_name']
            img_size = (img['height'], img['width'])
            if img_id in self.images:
                raise Exception('dulpicated image record')
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data['annotations']:
            img_id = bboxes['image_id']
            bbox = bboxes['bbox']
            bbox_label = self.label_map[bboxes['category_id']]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transforms = transforms

    #@property
    def labelnum(self):
        return len(self.label_info)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)

        img = Image.open(img_path).convert('RGB')

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        for (l, t, w, h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            bbox_size = (l / wtot, t / htot, r / wtot, b / htot)
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels = torch.tensor(bbox_labels)

        if self.transforms != None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transforms(img, (htot, wtot), bbox_sizes, bbox_labels)

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels


def split_coco_batch(batch: Batch, num_microbatches: int) -> Sequence[Batch]:  #type: ignore
    if not isinstance(batch, Sequence):
        raise ValueError(f'split_fn requires batch be a tuple of tensors, got {type(batch)}')
    img, img_id, img_size, bbox_sizes, bbox_labels = batch  #type: ignore
    nm = num_microbatches
    if isinstance(img, torch.Tensor) and isinstance(img_id, torch.Tensor):
        return list(
            zip(img.chunk(nm), img_id.chunk(nm), (img_size[i:i + nm] for i in range(0, len(img_size), nm)),
                bbox_sizes.chunk(nm), bbox_labels.chunk(nm)))  #type: ignore
    if isinstance(img, list) and isinstance(img_id, list) and isinstance(img_size, list) and isinstance(
            bbox_sizes, list) and isinstance(bbox_labels, list):
        return list(
            zip(
                [img[i::nm] for i in range(nm)],
                [img_id[i::nm] for i in range(nm)],
                [img_size[i::nm] for i in range(nm)],
                [bbox_sizes[i::nm] for i in range(nm)],
                [bbox_labels[i::nm] for i in range(nm)],
            ))  #type: ignore


def build_streaming_coco_dataloader(
    global_batch_size: int,
    remote: str,
    *,
    local: str = '/tmp/mds-cache/mds-coco',
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    input_size: int = 300,
    predownload: Optional[int] = 100_000,
    keep_zip: Optional[bool] = None,
    download_retry: int = 2,
    download_timeout: float = 60,
    validate_hash: Optional[str] = None,
    shuffle_seed: Optional[int] = None,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs: Dict[str, Any],
) -> DataSpec:
    """Builds a COCO streaming dataset

    Args:
        global_batch_size (int): Global batch size.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str, optional): Local filesystem directory where dataset is cached during operation.
            Defaults to ``'/tmp/mds-cache/mds-coco/```.
        split (str): Which split of the dataset to use. Either ['train', 'val']. Default:
            ``'train```.
        drop_last (bool, optional): whether to drop last samples. Default: ``True``.
        shuffle (bool, optional): whether to shuffle dataset. Defaults to ``True``.
        input_size (int): the size of the input image, keep this at `300` for SSD300. Default: ``300``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        **dataloader_kwargs (Any): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()

    # default boxes set for SSD300
    default_boxes = DefaultBoxes(fig_size=input_size,
                                 feat_size=[38, 19, 10, 5, 3, 1],
                                 steps=[8, 16, 32, 64, 100, 300],
                                 scales=[21, 45, 99, 153, 207, 261, 315],
                                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                 scale_xy=0.1,
                                 scale_wh=0.2)

    if split == 'train':
        transform = SSDTransformer(default_boxes, (input_size, input_size), val=False, num_cropping_iterations=1)
    else:
        transform = SSDTransformer(default_boxes, (input_size, input_size), val=True)

    try:
        from streaming.vision import StreamingCOCO
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e

    dataset = StreamingCOCO(
        local=local,
        remote=remote,
        split=split,
        shuffle=shuffle,
        transform=transform,
        predownload=predownload,
        keep_zip=keep_zip,
        download_retry=download_retry,
        download_timeout=download_timeout,
        validate_hash=validate_hash,
        shuffle_seed=shuffle_seed,
        num_canonical_nodes=num_canonical_nodes,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return DataSpec(dataloader=dataloader, split_batch=split_coco_batch)
