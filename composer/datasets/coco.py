# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""COCO (Common Objects in Context) dataset.

COCO is a large-scale object detection, segmentation, and captioning dataset. Please refer to the `COCO dataset
<https://cocodataset.org>`_ for more details.
"""
import json
import os
from io import BytesIO
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset

from composer.core import DataSpec
from composer.core.types import Batch
from composer.datasets.streaming import StreamingDataset
from composer.models.ssd.utils import DefaultBoxes, SSDTransformer
from composer.utils import dist

__all__ = ['COCODetection', 'StreamingCOCO', 'build_coco_detection_dataloader']


def build_coco_detection_dataloader(
    batch_size: int,
    datadir: str,
    *,
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    input_size: int = 300,
    **dataloader_kwargs,
):
    """Builds a COCO Detection dataloader with default transforms for SSD300.

    Args:
        batch_size (int): Batch size per device.
        datadir (str): Path to the data directory
        split (str): the dataset split to use either 'train', 'val', or 'test'. Default: ``'train```.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        input_size (int): the size of the input image, keep this at `300` for SSD300. Default: ``300``.
        **dataloader_kwargs (Any): Additional settings for the dataloader (e.g. num_workers, etc.)
    """

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


class StreamingCOCO(StreamingDataset, VisionDataset):
    """
    Implementation of the COCO dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def decode_img(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data)).convert('RGB')

    def decode_img_id(self, data: bytes) -> np.int64:
        return np.frombuffer(data, np.int64)[0]

    def decode_htot(self, data: bytes) -> np.int64:
        return np.frombuffer(data, np.int64)[0]

    def decode_wtot(self, data: bytes) -> np.int64:
        return np.frombuffer(data, np.int64)[0]

    def decode_bbox_sizes(self, data: bytes) -> torch.Tensor:
        arr = np.frombuffer(data, np.float32)
        arr = arr.reshape(-1, 4)
        return torch.tensor(arr)

    def decode_bbox_labels(self, data: bytes) -> torch.Tensor:
        arr = np.frombuffer(data, np.int64)
        return torch.tensor(arr)

    def __init__(self, remote: str, local: str, split: str, shuffle: bool, batch_size: Optional[int] = None) -> None:

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")

        # Build StreamingDataset
        decoders = {
            'img': self.decode_img,
            'img_id': self.decode_img_id,
            'htot': self.decode_htot,
            'wtot': self.decode_wtot,
            'bbox_sizes': self.decode_bbox_sizes,
            'bbox_labels': self.decode_bbox_labels,
        }
        super().__init__(remote=os.path.join(remote, split),
                         local=os.path.join(local, split),
                         shuffle=shuffle,
                         decoders=decoders,
                         batch_size=batch_size)

        # Define custom transforms
        # default boxes set for SSD300
        input_size = 300
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
        VisionDataset.__init__(self, root=local, transform=transform)

    def __getitem__(self, idx: int) -> Any:
        x = super().__getitem__(idx)
        img = x['img']
        img_id = x['img_id']
        htot = x['htot']
        wtot = x['wtot']
        bbox_sizes = x['bbox_sizes']
        bbox_labels = x['bbox_labels']
        assert self.transform is not None, 'transform set in __init__'
        img, (htot, wtot), bbox_sizes, bbox_labels = self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)
        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels
