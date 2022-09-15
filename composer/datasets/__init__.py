# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported datasets."""

from composer.datasets.ade20k import ADE20k, StreamingADE20k, build_ade20k_dataloader, build_synthetic_ade20k_dataloader
from composer.datasets.brats import PytTrain, PytVal
from composer.datasets.c4 import C4Dataset, StreamingC4
from composer.datasets.cifar import StreamingCIFAR10, build_cifar10_dataloader, build_synthetic_cifar10_dataloader
from composer.datasets.coco import COCODetection, StreamingCOCO
from composer.datasets.imagenet import (StreamingImageNet1k, build_ffcv_imagenet_dataloader, build_imagenet_dataloader,
                                        build_synthetic_imagenet_dataloader)
from composer.datasets.lm_dataset import build_lm_dataloader, build_synthetic_lm_dataloader
from composer.datasets.mnist import build_mnist_dataloader, build_synthetic_mnist_dataloader
from composer.datasets.synthetic import (SyntheticBatchPairDataset, SyntheticDataLabelType, SyntheticDataType,
                                         SyntheticPILDataset)

__all__ = [
    'ADE20k',
    'StreamingADE20k',
    'PytTrain',
    'PytVal',
    'C4Dataset',
    'StreamingC4',
    'StreamingCIFAR10',
    'COCODetection',
    'StreamingCOCO',
    'StreamingImageNet1k',
    'SyntheticBatchPairDataset',
    'SyntheticDataLabelType',
    'SyntheticDataType',
    'SyntheticPILDataset',
    'build_ade20k_dataloader',
    'build_cifar10_dataloader',
    'build_synthetic_ade20k_dataloader',
    'build_synthetic_cifar10_dataloader',
    'build_ffcv_imagenet_dataloader',
    'build_imagenet_dataloader',
    'build_synthetic_imagenet_dataloader',
    'build_mnist_dataloader',
    'build_synthetic_mnist_dataloader',
    'build_lm_dataloader',
    'build_synthetic_lm_dataloader',
]
