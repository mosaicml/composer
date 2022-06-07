# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Mapping between dataset names and corresponding HParams classes."""

from composer.datasets.ade20k import ADE20kDatasetHparams, StreamingADE20kHparams
from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.c4 import C4DatasetHparams
from composer.datasets.cifar import CIFAR10DatasetHparams
from composer.datasets.coco import COCODatasetHparams, StreamingCOCOHparams
from composer.datasets.glue import GLUEHparams
from composer.datasets.imagenet import ImagenetDatasetHparams, StreamingImageNet1kHparams
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.datasets.streaming_lm_datasets import StreamingLMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams

registry = {
    "ade20k": ADE20kDatasetHparams,
    "streaming_ade20k": StreamingADE20kHparams,
    "brats": BratsDatasetHparams,
    "imagenet": ImagenetDatasetHparams,
    "streaming_imagenet1k": StreamingImageNet1kHparams,
    "cifar10": CIFAR10DatasetHparams,
    "mnist": MNISTDatasetHparams,
    "lm": LMDatasetHparams,
    "streaming_lm": StreamingLMDatasetHparams,
    "glue": GLUEHparams,
    "coco": COCODatasetHparams,
    "streaming_coco": StreamingCOCOHparams,
    "c4": C4DatasetHparams,
}


def get_dataset_registry():
    """Returns a mapping between different supported datasets and their HParams classes that create an instance of the
    dataset. An example entry in the returned dictionary: ``"imagenet": ImagenetDatasetHparams``.

    Returns:
        Dict[str, DatasetHparams]: A dictionary of mapping.
    """

    return registry


__all__ = ["get_dataset_registry"]
