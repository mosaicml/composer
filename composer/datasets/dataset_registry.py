# Copyright 2021 MosaicML. All Rights Reserved.

from composer.datasets.ade20k import ADE20kDatasetHparams, ADE20kWebDatasetHparams
from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.c4 import C4DatasetHparams
from composer.datasets.cifar10 import (CIFAR10DatasetHparams, CIFAR10WebDatasetHparams, CIFAR20WebDatasetHparams,
                                       CIFAR100WebDatasetHparams)
from composer.datasets.coco import COCODatasetHparams
from composer.datasets.glue import GLUEHparams
from composer.datasets.imagenet import (ImagenetDatasetHparams, Imagenet1KWebDatasetHparams,
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
    'web_mnist': MNISTWebDatasetHparams,
    'web_cifar10': CIFAR10WebDatasetHparams,
    'web_cifar20': CIFAR20WebDatasetHparams,
    'web_cifar100': CIFAR100WebDatasetHparams,
    'web_tinyimagenet200': TinyImagenet200WebDatasetHparams,
    'web_imagenet1k': Imagenet1KWebDatasetHparams,
    'web_ade20k': ADE20kWebDatasetHparams,
}


def get_dataset_registry():
    return registry
