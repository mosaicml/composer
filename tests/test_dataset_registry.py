# Copyright 2021 MosaicML. All Rights Reserved.

import os
from typing import Callable, Dict, Type

import pytest

from composer.datasets.brats import BratsDatasetHparams
from composer.datasets.cifar10 import CIFAR10DatasetHparams
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.datasets.imagenet import ImagenetDatasetHparams
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams
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
            num_total_batches=1,
            download=False,
        ),
    BratsDatasetHparams:
        lambda: BratsDatasetHparams(
            is_train=False,
            datadir=os.path.join(ROOT_DATADIR, "01_2d"),
            num_total_batches=1,
        ),
    ImagenetDatasetHparams:
        lambda: ImagenetDatasetHparams(
            is_train=False,
            datadir=os.path.join(ROOT_DATADIR, "imagenet"),
            num_total_batches=1,
            crop_size=224,
            resize_size=-1,
        ),
    MNISTDatasetHparams:
        lambda: MNISTDatasetHparams(
            datadir=os.path.join(ROOT_DATADIR, "mnist"),
            num_total_batches=1,
            is_train=False,
            download=False,
        ),
    LMDatasetHparams:
        lambda: LMDatasetHparams(
            datadir=[os.path.join(ROOT_DATADIR, 'openwebtext_saved')],
            split='train',
            tokenizer_name='gpt2',
        )
}


@pytest.mark.parametrize("dataset_name", dataset_registry.keys())
def test_dataset(dataset_name: str, dummy_dataloader_hparams: DataloaderHparams,
                 request: pytest.FixtureRequest) -> None:
    hparams_cls = dataset_registry[dataset_name]
    hparams = default_required_fields[hparams_cls]()
    try:
        synthetic = hparams.synthetic
    except AttributeError:
        pytest.xfail(f"Dataset {dataset_name} does not support synthetic data")
        raise
    if synthetic is None:
        hparams.synthetic = hparams.get_synthetic_hparams_cls()()
    hparams.initialize_object(batch_size=1, dataloader_hparams=dummy_dataloader_hparams)
