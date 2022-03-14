# Copyright 2021 MosaicML. All Rights Reserved.

"""COCO (Common Objects in Context) dataset.

COCO is a large-scale object detection, segmentation, and captioning dataset. Please refer to the `COCO dataset
<https://cocodataset.org>`_ for more details.
"""
import json
import os
from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


from composer.core.types import Batch, DataSpec, List
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.models.ssd.utils import SSDTransformer, dboxes300_coco
from composer.utils import dist

__all__ = ["COCODatasetHparams", "COCODetection"]


@dataclass
class COCODatasetHparams(DatasetHparams):
    """Defines an instance of the COCO Dataset."""

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):

        if self.datadir is None:
            raise ValueError("datadir is required.")

        dboxes = dboxes300_coco()

        input_size = 300
        train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False, num_cropping_iterations=1)
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
        data = self.datadir

        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(data, "val2017")

        train_annotate = os.path.join(data, "annotations/instances_train2017.json")
        train_coco_root = os.path.join(data, "train2017")

        train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

        if self.is_train:
            return DataSpec(dataloader=dataloader_hparams.initialize_object(
                dataset=train_coco,
                batch_size=batch_size,
                sampler=dist.get_sampler(train_coco, drop_last=self.drop_last, shuffle=self.shuffle),
                drop_last=self.drop_last,
            ),
                            split_batch=split_dict_fn)
        else:
            return DataSpec(dataloader=dataloader_hparams.initialize_object(
                dataset=val_coco,
                drop_last=self.drop_last,
                batch_size=batch_size,
                sampler=None,
            ),
                            split_batch=split_dict_fn)


import torch.utils.data as data


class COCODetection(data.Dataset):
    """PyTorch Dataset for the COCO dataset.

    Args:
        img_folder (str): the path to the COCO folder.
        annotate_file (str): path to a file that contains image id, annotations (e.g., bounding boxes and object
            classes) etc.
        transform (torch.nn.Module): transformations to apply to the image.
    """

    def __init__(self, img_folder, annotate_file, transform=None):
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
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"], img["width"])
            if img_id in self.images:
                raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform

    #@property
    def labelnum(self):
        return len(self.label_info)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)

        img = Image.open(img_path).convert("RGB")

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

        if self.transform != None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels


def split_dict_fn(batch: Batch, num_microbatches: int) -> Sequence[Batch]:  #type: ignore
    if not isinstance(batch, Sequence):
        raise ValueError(f'split_fn requires batch be a tuple of tensors, got {type(batch)}')
    img, img_id, img_size, bbox_sizes, bbox_labels = batch  #type: ignore
    nm = num_microbatches
    if isinstance(img, torch.Tensor) and isinstance(img_id, torch.Tensor):
        return list(
            zip(img.chunk(nm), img_id.chunk(nm), (img_size[i:i + nm] for i in range(0, len(img_size), nm)),
                bbox_sizes.chunk(nm), bbox_labels.chunk(nm)))  #type: ignore
    if isinstance(img, List) and isinstance(img_id, List) and isinstance(img_size, List) and isinstance(
            bbox_sizes, List) and isinstance(bbox_labels, List):
        return list(
            zip(
                [img[i::nm] for i in range(nm)],
                [img_id[i::nm] for i in range(nm)],
                [img_size[i::nm] for i in range(nm)],
                [bbox_sizes[i::nm] for i in range(nm)],
                [bbox_labels[i::nm] for i in range(nm)],
            ))  #type: ignore
