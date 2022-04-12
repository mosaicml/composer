# Copyright 2021 MosaicML. All Rights Reserved.

"""Mapping between dataset names and corresponding HParams classes."""

from composer.datasets.ade20k import ADE20kDatasetHparams, ADE20kWebDatasetHparams, StreamingADE20kHparams
from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.c4 import C4DatasetHparams
from composer.datasets.cifar import (CIFAR10DatasetHparams, CIFAR10WebDatasetHparams, CIFAR20WebDatasetHparams,
                                     CIFAR100WebDatasetHparams, StreamingCIFAR10Hparams, StreamingCIFAR20Hparams,
                                     StreamingCIFAR100Hparams)
from composer.datasets.coco import COCODatasetHparams
from composer.datasets.glue import GLUEHparams
from composer.datasets.imagenet import (Imagenet1kWebDatasetHparams, ImagenetDatasetHparams, StreamingImagenet1kHparams,
                                        StreamingTinyImagenet200Hparams, TinyImagenet200WebDatasetHparams)
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams, MNISTWebDatasetHparams, StreamingMNISTHparams

registry = {
    "ade20k": ADE20kDatasetHparams,
    "brats": BratsDatasetHparams,
    "imagenet": ImagenetDatasetHparams,
    "cifar10": CIFAR10DatasetHparams,
    "mnist": MNISTDatasetHparams,
    "lm": LMDatasetHparams,
    "glue": GLUEHparams,
    "coco": COCODatasetHparams,
    "c4": C4DatasetHparams,
    'wds_mnist': MNISTWebDatasetHparams,
    'wds_cifar10': CIFAR10WebDatasetHparams,
    'wds_cifar20': CIFAR20WebDatasetHparams,
    'wds_cifar100': CIFAR100WebDatasetHparams,
    'wds_tinyimagenet200': TinyImagenet200WebDatasetHparams,
    'wds_imagenet1k': Imagenet1kWebDatasetHparams,
    'wds_ade20k': ADE20kWebDatasetHparams,
    'streaming_mnist': StreamingMNISTHparams,
    'streaming_cifar10': StreamingCIFAR10Hparams,
    'streaming_cifar20': StreamingCIFAR20Hparams,
    'streaming_cifar100': StreamingCIFAR100Hparams,
    'streaming_tinyimagenet200': StreamingTinyImagenet200Hparams,
    'streaming_imagenet1k': StreamingImagenet1kHparams,
    'streaming_ade20k': StreamingADE20kHparams,
}


def get_dataset_registry():
    """Returns a mapping between different supported datasets and their HParams classes that create an instance of the
    dataset. An example entry in the returned dictionary: ``"imagenet": ImagenetDatasetHparams``.

    Returns:
        Dict[str, DatasetHparams]: A dictionary of mapping.
    """

    return registry


__all__ = ["get_dataset_registry"]
