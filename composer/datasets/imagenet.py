# Copyright 2021 MosaicML. All Rights Reserved.

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import yahp as hp
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from composer.core.types import Batch, Tensor
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DataloaderSpec, DatasetHparams, SyntheticHparamsMixin
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist


class TransformationFn:

    def __init__(self) -> None:
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None

    def __call__(self, batch: Batch):
        xs, ys = batch
        assert isinstance(xs, Tensor)
        assert isinstance(ys, Tensor)
        device = xs.device

        if self.mean is None:
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=device)
            self.mean = self.mean.view(1, 3, 1, 1)
        if self.std is None:
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=device)
            self.std = self.std.view(1, 3, 1, 1)

        xs = xs.float()
        xs = xs.sub_(self.mean).div_(self.std)
        return xs, ys


def fast_collate(batch: List[Tuple[Image.Image, Tensor]], memory_format: torch.memory_format = torch.contiguous_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2).copy()
        if nump_array.shape[0] != 3:
            assert nump_array.shape[0] == 1, "unexpected shape"
            nump_array = np.resize(nump_array, (3, h, w))
        assert tuple(tensor.shape)[1:] == nump_array.shape, "shape mistmatch"

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


@dataclass
class ImagenetDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ImageNet dataset for image classification.
    
    Parameters:
        resize_size (int, optional): The resize size to use. Defaults to -1 to not resize.
        crop size (int): The crop size to use.
    """
    resize_size: int = hp.optional("resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("crop size", default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataloaderSpec:

        if self.use_synthetic:
            total_dataset_size = 1_281_167 if self.is_train else 50_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, self.crop_size, self.crop_size],
                num_classes=1000,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
            collate_fn = None
            device_transform_fn = None
        else:

            if self.is_train:
                # include fixed-size resize before RandomResizedCrop in training only
                # if requested (by specifying a size > 0)
                train_resize_size = self.resize_size
                train_transforms: List[torch.nn.Module] = []
                if train_resize_size > 0:
                    train_transforms.append(transforms.Resize(train_resize_size))
                # always include RandomResizedCrop and RandomHorizontalFlip
                train_transforms += [
                    transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                    transforms.RandomHorizontalFlip()
                ]
                transformation = transforms.Compose(train_transforms)
                split = "train"
            else:
                transformation = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                ])
                split = "val"

            device_transform_fn = TransformationFn()
            collate_fn = fast_collate

            if self.datadir is None:
                raise ValueError("datadir must be specified is self.synthetic is False")
            dataset = ImageFolder(os.path.join(self.datadir, split), transformation)
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return DataloaderSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                              device_transform_fn=device_transform_fn)
