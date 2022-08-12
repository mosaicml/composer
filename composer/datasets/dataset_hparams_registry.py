# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Mapping between dataset names and corresponding HParams classes.

Attributes:
    dataset_registry ( Dict[str, DatasetHparams]): The dataset registry.
"""

from composer.datasets.ade20k_hparams import ADE20kDatasetHparams, StreamingADE20kHparams
from composer.datasets.brats_hparams import BratsDatasetHparams
from composer.datasets.c4_hparams import C4DatasetHparams, StreamingC4Hparams
from composer.datasets.cifar_hparams import CIFAR10DatasetHparams, StreamingCIFAR10Hparams
from composer.datasets.coco_hparams import COCODatasetHparams, StreamingCOCOHparams
from composer.datasets.glue_hparams import GLUEHparams
from composer.datasets.imagenet_hparams import ImagenetDatasetHparams, StreamingImageNet1kHparams
from composer.datasets.lm_dataset_hparams import LMDatasetHparams
from composer.datasets.mnist_hparams import MNISTDatasetHparams

__all__ = ['dataset_registry']

dataset_registry = {
    'ade20k': ADE20kDatasetHparams,
    'streaming_ade20k': StreamingADE20kHparams,
    'brats': BratsDatasetHparams,
    'imagenet': ImagenetDatasetHparams,
    'streaming_imagenet1k': StreamingImageNet1kHparams,
    'cifar10': CIFAR10DatasetHparams,
    'streaming_cifar10': StreamingCIFAR10Hparams,
    'mnist': MNISTDatasetHparams,
    'lm': LMDatasetHparams,
    'glue': GLUEHparams,
    'coco': COCODatasetHparams,
    'streaming_coco': StreamingCOCOHparams,
    'c4': C4DatasetHparams,
    'streaming_c4': StreamingC4Hparams,
}
