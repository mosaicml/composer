# Copyright 2021 MosaicML. All Rights Reserved.

from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.cifar10 import CIFAR10DatasetHparams
from composer.datasets.imagenet import ImagenetDatasetHparams
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams
from composer.datasets.synthetic import SyntheticDatasetHparams

registry = {
    "brats": BratsDatasetHparams,
    "imagenet": ImagenetDatasetHparams,
    "cifar10": CIFAR10DatasetHparams,
    "synthetic": SyntheticDatasetHparams,
    "mnist": MNISTDatasetHparams,
    "lm": LMDatasetHparams,
}


def get_dataset_registry():
    return registry