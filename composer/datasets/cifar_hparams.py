# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR image classification dataset hyperparameters.

The CIFAR datasets are a collection of labeled 32x32 colour images. Please refer to the `CIFAR dataset
<https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more details.
"""

import logging
import os
import textwrap
from dataclasses import dataclass
from typing import List, Optional

import torch
import yahp as hp
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.datasets.cifar import StreamingCIFAR10
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.utils import dist

__all__ = ['CIFAR10DatasetHparams', 'StreamingCIFAR10Hparams']

log = logging.getLogger(__name__)


@dataclass
class CIFAR10DatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
    """
    download: bool = hp.optional('whether to download the dataset, if needed', default=True)
    use_ffcv: bool = hp.optional('whether to use ffcv for faster dataloading', default=False)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default='/tmp')
    ffcv_dest: str = hp.optional('<file>.ffcv file that has dataset samples', default='cifar_train.ffcv')
    ffcv_write_dataset: bool = hp.optional("Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist",
                                           default=False)

    is_train: bool = hp.optional('Whether to load the training data (the default) or validation data.', default=True)
    datadir: Optional[str] = hp.optional('The path to the data directory', default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):

        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                import ffcv
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
                from ffcv.pipeline.operation import Operation
            except ImportError:
                raise ImportError(
                    textwrap.dedent("""\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""))

            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            'datadir is required if use_synthetic is False and ffcv_write_dataset is True.')
                    ds = CIFAR10(
                        self.datadir,
                        train=self.is_train,
                        download=self.download,
                    )

                    write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            if not os.path.exists(dataset_filepath):
                raise ValueError(
                    f'Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}.'
                )

            # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
            # ToTensor does the normalization for PyTorch dataloaders.
            cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
            cifar10_std_ffcv = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend([
                    ffcv.transforms.RandomHorizontalFlip(),
                    ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
                    ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
                ])
            # Common transforms for train and test
            image_pipeline.extend([
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                ffcv.transforms.Convert(torch.float32),
                transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
            ])

            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=False,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )
        else:
            if self.datadir is None:
                raise ValueError('datadir is required if use_synthetic is False')

            cifar10_mean = 0.4914, 0.4822, 0.4465
            cifar10_std = 0.247, 0.243, 0.261

            if self.is_train:
                transformation = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std),
                ])
            else:
                transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std),
                ])

            with dist.run_local_rank_zero_first():
                dataset = CIFAR10(
                    self.datadir,
                    train=self.is_train,
                    download=dist.get_local_rank() == 0 and self.download,
                    transform=transformation,
                )

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)


@dataclass
class StreamingCIFAR10Hparams(DatasetHparams):
    """Streaming CIFAR10 hyperparameters.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-cifar10/mds/1/'``
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-cifar10/'``
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train'``.
    """

    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-cifar10/mds/1/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-cifar10/')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val']", default='train')

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader:
        dataset = StreamingCIFAR10(remote=self.remote,
                                   local=self.local,
                                   split=self.split,
                                   shuffle=self.shuffle,
                                   batch_size=batch_size)
        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    drop_last=self.drop_last)
