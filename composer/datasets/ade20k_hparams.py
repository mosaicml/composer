# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for more details about this
dataset.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.transforms.functional as TF
import yahp as hp
from torchvision import transforms

from composer.core import DataSpec
from composer.datasets.ade20k import (IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD, ADE20k, PadToSize,
                                      PhotometricDistoration, RandomCropPair, RandomHFlipPair, RandomResizePair,
                                      StreamingADE20k)
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist

__all__ = ['ADE20kDatasetHparams', 'StreamingADE20kHparams']


@dataclass
class ADE20kDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ADE20k dataset for semantic segmentation from a local disk.

    Args:
        split (str): the dataset split to use either 'train', 'val', or 'test'. Default: ``'train```.
        base_size (int): initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): the minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): the maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): the final size of the image and target. Default: ``512``.
        ignore_background (bool): if true, ignore the background class when calculating the training loss.
            Default: ``true``.
        datadir (str): The path to the data directory.
    """

    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val', 'test']", default='train')
    base_size: int = hp.optional('Initial size of the image and target before other augmentations', default=512)
    min_resize_scale: float = hp.optional('Minimum value that the image and target can be scaled', default=0.5)
    max_resize_scale: float = hp.optional('Maximum value that the image and target can be scaled', default=2.0)
    final_size: int = hp.optional('Final size of the image and target', default=512)
    ignore_background: bool = hp.optional('If true, ignore the background class in training loss', default=True)

    datadir: Optional[str] = hp.optional('The path to the data directory', default=None)

    def validate(self):
        if self.datadir is None and not self.use_synthetic:
            raise ValueError('datadir must specify the path to the ADE20k dataset.')

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"split value {self.split} must be one of ['train', 'val', 'test'].")

        if self.base_size <= 0:
            raise ValueError('base_size cannot be zero or negative.')

        if self.min_resize_scale <= 0:
            raise ValueError('min_resize_scale cannot be zero or negative')

        if self.max_resize_scale < self.min_resize_scale:
            raise ValueError('max_resize_scale cannot be less than min_resize_scale')

    def initialize_object(self, batch_size, dataloader_hparams) -> DataSpec:
        self.validate()

        if self.use_synthetic:
            if self.split == 'train':
                total_dataset_size = 20_206
            elif self.split == 'val':
                total_dataset_size = 2_000
            else:
                total_dataset_size = 3_352

            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, self.final_size, self.final_size],
                label_shape=[self.final_size, self.final_size],
                num_classes=150,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
            collate_fn = None
            device_transform_fn = None

        else:
            # Define data transformations based on data split
            if self.split == 'train':
                both_transforms = torch.nn.Sequential(
                    RandomResizePair(min_scale=self.min_resize_scale,
                                     max_scale=self.max_resize_scale,
                                     base_size=(self.base_size, self.base_size)),
                    RandomCropPair(
                        crop_size=(self.final_size, self.final_size),
                        class_max_percent=0.75,
                        num_retry=10,
                    ),
                    RandomHFlipPair(),
                )

                # Photometric distoration values come from mmsegmentation:
                # https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L861
                r_mean, g_mean, b_mean = IMAGENET_CHANNEL_MEAN
                image_transforms = torch.nn.Sequential(
                    PhotometricDistoration(brightness=32. / 255, contrast=0.5, saturation=0.5, hue=18. / 255),
                    PadToSize(size=(self.final_size, self.final_size), fill=(int(r_mean), int(g_mean), int(b_mean))))

                target_transforms = PadToSize(size=(self.final_size, self.final_size), fill=0)
            else:
                both_transforms = None
                image_transforms = transforms.Resize(size=(self.final_size, self.final_size),
                                                     interpolation=TF.InterpolationMode.BILINEAR)
                target_transforms = transforms.Resize(size=(self.final_size, self.final_size),
                                                      interpolation=TF.InterpolationMode.NEAREST)
            collate_fn = pil_image_collate
            device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                                  std=IMAGENET_CHANNEL_STD,
                                                  ignore_background=self.ignore_background)

            # Add check to avoid type ignore below
            if self.datadir is None:
                raise ValueError('datadir must specify the path to the ADE20k dataset.')

            dataset = ADE20k(datadir=self.datadir,
                             split=self.split,
                             both_transforms=both_transforms,
                             image_transforms=image_transforms,
                             target_transforms=target_transforms)
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(dataset=dataset,
                                                                        batch_size=batch_size,
                                                                        sampler=sampler,
                                                                        collate_fn=collate_fn,
                                                                        drop_last=self.drop_last),
                        device_transforms=device_transform_fn)


@dataclass
class StreamingADE20kHparams(DatasetHparams):
    """DatasetHparams for creating an instance of StreamingADE20k.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-ade20k/mds/1/```
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-ade20k/```
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
        base_size (int): initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): the minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): the maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): the final size of the image and target. Default: ``512``.
        ignore_background (bool): if true, ignore the background class when calculating the training loss.
            Default: ``true``.
    """

    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-ade20k/mds/1/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-ade20k/')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val']", default='train')
    base_size: int = hp.optional('Initial size of the image and target before other augmentations', default=512)
    min_resize_scale: float = hp.optional('Minimum value that the image and target can be scaled', default=0.5)
    max_resize_scale: float = hp.optional('Maximum value that the image and target can be scaled', default=2.0)
    final_size: int = hp.optional('Final size of the image and target', default=512)
    ignore_background: bool = hp.optional('If true, ignore the background class in training loss', default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        dataset = StreamingADE20k(remote=self.remote,
                                  local=self.local,
                                  split=self.split,
                                  shuffle=self.shuffle,
                                  base_size=self.base_size,
                                  min_resize_scale=self.min_resize_scale,
                                  max_resize_scale=self.max_resize_scale,
                                  final_size=self.final_size,
                                  batch_size=batch_size)
        collate_fn = pil_image_collate
        device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                              std=IMAGENET_CHANNEL_STD,
                                              ignore_background=self.ignore_background)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(dataset=dataset,
                                                                        batch_size=batch_size,
                                                                        sampler=None,
                                                                        collate_fn=collate_fn,
                                                                        drop_last=self.drop_last),
                        device_transforms=device_transform_fn)
