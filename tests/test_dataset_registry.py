# Copyright 2021 MosaicML. All Rights Reserved.

import os
from typing import Callable, Dict, Type

import pytest

from composer.core import DataSpec
from composer.datasets import (BratsDatasetHparams, CIFAR10DatasetHparams, DataloaderHparams, DatasetHparams,
                               ImagenetDatasetHparams, LMDatasetHparams, MNISTDatasetHparams, SyntheticHparamsMixin)
from composer.trainer.trainer_hparams import dataset_registry

# for testing, we provide values for required hparams fields
# to initialize test hparams objects
default_required_fields: Dict[Type[DatasetHparams], Callable[[], DatasetHparams]] = {
    #  hparams with empty dicts have no required fields
    CIFAR10DatasetHparams: lambda: CIFAR10DatasetHparams(
        is_train=False,
        download=False,
    ),
    BratsDatasetHparams: lambda: BratsDatasetHparams(is_train=False,),
    ImagenetDatasetHparams: lambda: ImagenetDatasetHparams(
        is_train=False,
        crop_size=224,
        resize_size=-1,
    ),
    MNISTDatasetHparams: lambda: MNISTDatasetHparams(
        is_train=False,
        download=False,
    ),
    LMDatasetHparams: lambda: LMDatasetHparams(
        datadir=["hello"],
        split='train',
        tokenizer_name='gpt2',
    )
}


@pytest.mark.parametrize("dataset_name", dataset_registry.keys())
def test_dataset(dataset_name: str, dummy_dataloader_hparams: DataloaderHparams) -> None:
    hparams_cls = dataset_registry[dataset_name]
    hparams = default_required_fields[hparams_cls]()
    if not isinstance(hparams, SyntheticHparamsMixin):
        pytest.xfail(f"{hparams.__class__.__name__} does not support synthetic data")

    assert isinstance(hparams, SyntheticHparamsMixin)

    hparams.use_synthetic = True

    hparams.initialize_object(batch_size=1, dataloader_hparams=dummy_dataloader_hparams)


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.timeout(10)
def test_mnist_real_dataset(world_size: int, dummy_dataloader_hparams: DataloaderHparams, tmpdir: str):
    # only test mnist since it has a small validation dataset
    hparams_cls = dataset_registry["mnist"]
    hparams = default_required_fields[hparams_cls]()
    assert isinstance(hparams, MNISTDatasetHparams)
    hparams.download = True
    hparams.datadir = os.path.join(tmpdir, "mnist_data")
    hparams.is_train = False
    hparams.use_synthetic = False
    batch_size = 10
    device_batch_size = batch_size // world_size
    dataloader = hparams.initialize_object(batch_size=device_batch_size, dataloader_hparams=dummy_dataloader_hparams)
    assert not isinstance(dataloader, DataSpec)
    assert len(dataloader) == 10_000 // batch_size  # mnist has 10_000 validation images
