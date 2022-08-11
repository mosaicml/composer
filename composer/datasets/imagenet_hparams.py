# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classification dataset hyperparameters.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet 2012 Classification
Dataset <http://image-net.org/>`_ for more details.
"""

import os
import textwrap
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data
import yahp as hp
from torchvision import transforms
from torchvision.datasets import ImageFolder

from composer.core import DataSpec
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.ffcv_utils import ffcv_monkey_patches, write_ffcv_dataset
from composer.datasets.imagenet import StreamingImageNet1k
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist

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
        elif self.use_ffcv:
            try:
                import ffcv
                from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder, RandomResizedCropRGBImageDecoder
                from ffcv.pipeline.operation import Operation
            except ImportError:
                raise ImportError(
                    textwrap.dedent("""\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""))

            if self.is_train:
                split = 'train'
            else:
                split = 'val'
            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            'datadir is required if use_synthetic is False and ffcv_write_dataset is True.')
                    ds = ImageFolder(os.path.join(self.datadir, split))
                    write_ffcv_dataset(dataset=ds,
                                       write_path=dataset_filepath,
                                       max_resolution=500,
                                       num_workers=dataloader_hparams.num_workers,
                                       compress_probability=0.50,
                                       jpeg_quality=90)
                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            this_device = torch.device(f'cuda:{dist.get_local_rank()}')
            label_pipeline: List[Operation] = [
                IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
                ffcv.transforms.ToDevice(this_device, non_blocking=True)
            ]
            image_pipeline: List[Operation] = []
            if self.is_train:
                image_pipeline.extend([
                    RandomResizedCropRGBImageDecoder((self.crop_size, self.crop_size)),
                    ffcv.transforms.RandomHorizontalFlip()
                ])
                dtype = np.float16
            else:
                ratio = self.crop_size / self.resize_size if self.resize_size > 0 else 1.0
                image_pipeline.extend([CenterCropRGBImageDecoder((self.crop_size, self.crop_size), ratio=ratio)])
                dtype = np.float32
            # Common transforms for train and test
            if self.ffcv_cpu_only:
                image_pipeline.extend([
                    ffcv.transforms.NormalizeImage(np.array(IMAGENET_CHANNEL_MEAN), np.array(IMAGENET_CHANNEL_STD),
                                                   dtype),
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToTorchImage(),
                ])
            else:
                image_pipeline.extend([
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToDevice(this_device, non_blocking=True),
                    ffcv.transforms.ToTorchImage(),
                    ffcv.transforms.NormalizeImage(np.array(IMAGENET_CHANNEL_MEAN), np.array(IMAGENET_CHANNEL_STD),
                                                   dtype),
                ])

            is_distributed = dist.get_world_size() > 1

            ffcv_monkey_patches()
            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=is_distributed,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )

        else:

            if self.is_train:
                # include fixed-size resize before RandomResizedCrop in training only
                # if requested (by specifying a size > 0)
                train_transforms: List[torch.nn.Module] = []
                if self.resize_size > 0:
                    train_transforms.append(transforms.Resize(self.resize_size))
                # always include RandomResizedCrop and RandomHorizontalFlip
                train_transforms += [
                    transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                    transforms.RandomHorizontalFlip()
                ]
                transformation = transforms.Compose(train_transforms)
                split = 'train'
            else:
                val_transforms: List[torch.nn.Module] = []
                if self.resize_size > 0:
                    val_transforms.append(transforms.Resize(self.resize_size))
                val_transforms.append(transforms.CenterCrop(self.crop_size))
                transformation = transforms.Compose(val_transforms)
                split = 'val'

            device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)
            collate_fn = pil_image_collate

            if self.datadir is None:
                raise ValueError('datadir must be specified if self.synthetic is False')
            dataset = ImageFolder(os.path.join(self.datadir, split), transformation)
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                        device_transforms=device_transform_fn)


@dataclass
class StreamingImageNet1kHparams(DatasetHparams):
    """DatasetHparams for creating an instance of StreamingImageNet1k.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-imagenet1k/mds/1/```
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-imagenet1k/```
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
        resize_size (int, optional): The resize size to use. Use -1 to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
    """

    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-imagenet1k/mds/1/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-imagenet1k/')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val']", default='train')
    resize_size: int = hp.optional('Resize size. Set to -1 to not resize', default=-1)
    crop_size: int = hp.optional('Crop size', default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        dataset = StreamingImageNet1k(remote=self.remote,
                                      local=self.local,
                                      split=self.split,
                                      shuffle=self.shuffle,
                                      resize_size=self.resize_size,
                                      crop_size=self.crop_size,
                                      batch_size=batch_size)
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
