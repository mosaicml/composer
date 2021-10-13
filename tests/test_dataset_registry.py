# Copyright 2021 MosaicML. All Rights Reserved.

import os
from typing import Callable, Dict, Type

import pytest

from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.cifar10 import CIFAR10DatasetHparams
from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.datasets.imagenet import ImagenetDatasetHparams
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams
from composer.datasets.synthetic import SyntheticDatasetHparams
from composer.trainer.trainer_hparams import dataset_registry

ROOT_DATADIR = os.environ.get('MOSAICML_DATASET_DIR', "/tmp")

# for testing, we provide values for required hparams fields
# to initialize test hparams objects
default_required_fields: Dict[Type[DatasetHparams], Callable[[], DatasetHparams]] = {
    #  hparams with empty dicts have no required fields
    CIFAR10DatasetHparams:
        lambda: CIFAR10DatasetHparams(
            is_train=False,
            datadir=os.path.join(ROOT_DATADIR, "cifar10"),
            download=False,
        ),
    BratsDatasetHparams:
        lambda: BratsDatasetHparams(
            is_train=False,
            datadir=os.path.join(ROOT_DATADIR, "01_2d"),
            download=False,
        ),
    ImagenetDatasetHparams:
        lambda: ImagenetDatasetHparams(
            is_train=False,
            datadir=os.path.join(ROOT_DATADIR, "imagenet"),
            crop_size=224,
            resize_size=-1,
        ),
    MNISTDatasetHparams:
        lambda: MNISTDatasetHparams(
            datadir=os.path.join(ROOT_DATADIR, "mnist"),
            is_train=False,
            download=False,
        ),
    SyntheticDatasetHparams:
        lambda: SyntheticDatasetHparams(
            num_classes=100,
            shape=[256, 256],
            one_hot=False,
            sample_pool_size=20,
            device="cpu",
        ),
    LMDatasetHparams:
        lambda: LMDatasetHparams(
            datadir=[os.path.join(ROOT_DATADIR, 'openwebtext_saved')],
            split='train',
            tokenizer_name='gpt2',
        )
}


@pytest.mark.parametrize("dataset_name", dataset_registry.keys())
def test_dataset(dataset_name: str, request: pytest.FixtureRequest) -> None:
    hparams_cls = dataset_registry[dataset_name]
    hparams = default_required_fields[hparams_cls]()
    if dataset_name != "synthetic":
        request.applymarker(pytest.mark.xfail())
    dataloader_spec = hparams.initialize_object()
    assert isinstance(dataloader_spec, DataloaderSpec)
