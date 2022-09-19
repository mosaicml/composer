# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classification dataset hyperparameters.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet 2012 Classification
Dataset <http://image-net.org/>`_ for more details.
"""

import os
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp
from torchvision import transforms

from composer.core import DataSpec
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.imagenet import (StreamingImageNet1k, build_ffcv_imagenet_dataloader, build_imagenet_dataloader,
                                        build_synthetic_imagenet_dataloader, write_ffcv_imagenet)
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import warn_streaming_dataset_deprecation
from composer.utils.import_helpers import MissingConditionalImportError

# ImageNet normalization values from torchvision: https://pytorch.org/vision/stable/models.html
IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)

__all__ = ['ImagenetDatasetHparams', 'StreamingImageNet1kHparams']


@dataclass
class ImagenetDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ImageNet dataset for image classification.

    Args:
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"imagenet_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
    """
    resize_size: int = hp.optional('resize size. Set to -1 to not resize', default=-1)
    crop_size: int = hp.optional('crop size', default=224)
    use_ffcv: bool = hp.optional('whether to use ffcv for faster dataloading', default=False)
    ffcv_cpu_only: bool = hp.optional('Use cpu for all transformations.', default=False)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default='/tmp')
    ffcv_dest: str = hp.optional('<file>.ffcv file that has dataset samples', default='imagenet_train.ffcv')
    ffcv_write_dataset: bool = hp.optional("Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist",
                                           default=False)
    is_train: bool = hp.optional('Whether to load the training data (the default) or validation data.', default=True)
    datadir: Optional[str] = hp.optional('The path to the data directory', default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:

        if self.use_synthetic:
            return build_synthetic_imagenet_dataloader(
                batch_size=batch_size,
                num_unique_samples=self.synthetic_num_unique_samples,
                memory_format=self.synthetic_memory_format,
                device=self.synthetic_device,
                is_train=self.is_train,
                crop_size=self.crop_size,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
                **asdict(dataloader_hparams),
            )
        elif self.use_ffcv:

            ffcv_dir = os.path.join(self.ffcv_dir, self.ffcv_dest)

            if self.ffcv_write_dataset:
                if self.datadir is None:
                    raise ValueError('datadir must be provided when writing FFCV dataset.')

                write_ffcv_imagenet(
                    datadir=self.datadir,
                    savedir=ffcv_dir,
                    split='train' if self.is_train else 'val',
                    num_workers=dataloader_hparams.num_workers,
                )

            return build_ffcv_imagenet_dataloader(
                datadir=ffcv_dir,
                batch_size=batch_size,
                is_train=self.is_train,
                resize_size=self.resize_size,
                crop_size=self.crop_size,
                cpu_only=self.ffcv_cpu_only,
                drop_last=self.drop_last,
                prefetch_factor=dataloader_hparams.prefetch_factor,
                num_workers=dataloader_hparams.num_workers,
            )

        else:
            if self.datadir is None:
                raise ValueError('datadir must be specified if self.synthetic is False')

            return build_imagenet_dataloader(
                datadir=self.datadir,
                batch_size=batch_size,
                is_train=self.is_train,
                resize_size=self.resize_size,
                crop_size=self.crop_size,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
                **asdict(dataloader_hparams),
            )


@dataclass
class StreamingImageNet1kHparams(DatasetHparams):
    """DatasetHparams for creating an instance of StreamingImageNet1k.

    Args:
        version (int): Which version of streaming to use. Default: ``1``.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-imagenet1k/mds/1/```
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-imagenet1k/```
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
        resize_size (int, optional): The resize size to use. Use -1 to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
    """

    version: int = hp.optional('Version of streaming (1 or 2)', default=1)
    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-imagenet1k/mds/1/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-imagenet1k/')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val']", default='train')
    resize_size: int = hp.optional('Resize size. Set to -1 to not resize', default=-1)
    crop_size: int = hp.optional('Crop size', default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        if self.version == 1:
            warn_streaming_dataset_deprecation(old_version=self.version, new_version=2)
            dataset = StreamingImageNet1k(remote=self.remote,
                                          local=self.local,
                                          split=self.split,
                                          shuffle=self.shuffle,
                                          resize_size=self.resize_size,
                                          crop_size=self.crop_size,
                                          batch_size=batch_size)
        elif self.version == 2:
            try:
                from streaming.vision import ImageNet
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='streaming',
                                                    conda_package='mosaicml-streaming') from e
            transform = []
            if self.split == 'train':
                # include fixed-size resize before RandomResizedCrop in training only
                # if requested (by specifying a size > 0)
                if self.resize_size > 0:
                    transform.append(transforms.Resize(self.resize_size))
                # always include RandomResizedCrop and RandomHorizontalFlip
                transform += [
                    transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                    transforms.RandomHorizontalFlip()
                ]
            else:
                if self.resize_size > 0:
                    transform.append(transforms.Resize(self.resize_size))
                transform.append(transforms.CenterCrop(self.crop_size))
            transform.append(lambda image: image.convert('RGB'))
            transform = transforms.Compose(transform)
            dataset = ImageNet(local=self.local,
                               remote=self.remote,
                               split=self.split,
                               shuffle=self.shuffle,
                               transform=transform,
                               batch_size=batch_size)
        else:
            raise ValueError(f'Invalid streaming version: {self.version}')
        collate_fn = pil_image_collate
        device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                        device_transforms=device_transform_fn)
