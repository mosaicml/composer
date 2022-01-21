# Copyright 2021 MosaicML. All Rights Reserved.

from composer.datasets.ade20k import ADE20kDatasetHparams
from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.cifar10 import CIFAR10DatasetHparams
from composer.datasets.glue import GLUEHparams
from composer.datasets.imagenet import ImagenetDatasetHparams
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams

registry = {
    "ade20k": ADE20kDatasetHparams,
    "brats": BratsDatasetHparams,
    "imagenet": ImagenetDatasetHparams,
    "cifar10": CIFAR10DatasetHparams,
    "mnist": MNISTDatasetHparams,
    "lm": LMDatasetHparams,
    "glue": GLUEHparams
}


def get_dataset_registry():
    return registry
