# Copyright 2021 MosaicML. All Rights Reserved.

"""Mapping between dataset names and corresponding HParams classes."""

from composer.datasets.ade20k import ADE20kDatasetHparams, ADE20kWebDatasetHparams
from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.c4 import C4DatasetHparams
from composer.datasets.cifar import (CIFAR10DatasetHparams, CIFAR10WebDatasetHparams, CIFAR20WebDatasetHparams,
                                     CIFAR100WebDatasetHparams)
from composer.datasets.coco import COCODatasetHparams
from composer.datasets.glue import GLUEHparams
from composer.datasets.imagenet import (Imagenet1kWebDatasetHparams, ImagenetDatasetHparams,
                                        TinyImagenet200WebDatasetHparams)
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams, MNISTWebDatasetHparams

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
}


def get_dataset_registry():
    """Returns a mapping between different supported datasets and their HParams classes that create an instance of the
    dataset. An example entry in the returned dictionary: ``"imagenet": ImagenetDatasetHparams``.

    Returns:
        Dict[str, DatasetHparams]: A dictionary of mapping.
    """

    return registry


__all__ = ["get_dataset_registry"]
