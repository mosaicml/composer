# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
import os
from typing import Callable, Dict, Optional, Type

import pytest

from composer.datasets import (BratsDatasetHparams, CIFAR10DatasetHparams, DataloaderHparams, DataloaderSpec,
                               DatasetHparams, ImagenetDatasetHparams, LMDatasetHparams, MNISTDatasetHparams,
                               SyntheticHparamsMixin)
from composer.trainer.trainer_hparams import dataset_registry

# for testing, we provide values for required hparams fields
# to initialize test hparams objects
default_required_fields: Dict[Type[DatasetHparams], Callable[[], DatasetHparams]] = {
    #  hparams with empty dicts have no required fields
    CIFAR10DatasetHparams:
        lambda: CIFAR10DatasetHparams(
            is_train=False,
            subset_num_batches=1,
            download=False,
        ),
    BratsDatasetHparams:
        lambda: BratsDatasetHparams(
            is_train=False,
            subset_num_batches=2,
        ),
    ImagenetDatasetHparams:
        lambda: ImagenetDatasetHparams(
            is_train=False,
            subset_num_batches=3,
            crop_size=224,
            resize_size=-1,
        ),
    MNISTDatasetHparams:
        lambda: MNISTDatasetHparams(
            subset_num_batches=4,
            is_train=False,
            download=False,
        ),
    LMDatasetHparams:
        lambda: LMDatasetHparams(
            datadir=["hello"],
            split='train',
            tokenizer_name='gpt2',
            subset_num_batches=5,
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

    dataloader = hparams.initialize_object(batch_size=1, dataloader_hparams=dummy_dataloader_hparams)
    if isinstance(dataloader, DataloaderSpec):
        dataloader = dataloader.dataloader
    assert len(dataloader) == hparams.subset_num_batches


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize(
    "synthetic",
    [True, pytest.param(False, marks=pytest.mark.timeout(30))],
)
@pytest.mark.parametrize("subset_num_batches", [None, 7])
def test_subset_num_batches(world_size: int, synthetic: bool, dummy_dataloader_hparams: DataloaderHparams, tmpdir: str,
                            subset_num_batches: Optional[int]):
    # only test mnist since it has a small validation dataset
    if subset_num_batches is None and synthetic is True:
        pytest.skip("Skipping test since synthetic is True and subset_num_batches is None")
    hparams_cls = dataset_registry["mnist"]
    hparams = default_required_fields[hparams_cls]()
    assert isinstance(hparams, MNISTDatasetHparams)
    hparams.download = not synthetic
    hparams.datadir = os.path.join(tmpdir, "mnist_data")
    hparams.is_train = False
    hparams.use_synthetic = synthetic
    hparams.subset_num_batches = subset_num_batches
    batch_size = 10
    device_batch_size = batch_size // world_size
    dataloader = hparams.initialize_object(batch_size=device_batch_size, dataloader_hparams=dummy_dataloader_hparams)
    assert not isinstance(dataloader, DataloaderSpec)

    if subset_num_batches is None:
        num_total_samples = 10000
        num_device_samples = (num_total_samples // world_size)
        subset_num_batches = num_device_samples // device_batch_size

    assert isinstance(dataloader.sampler, collections.abc.Sized)
    assert len(dataloader.sampler) == subset_num_batches * device_batch_size
    assert len(dataloader) == subset_num_batches

    if subset_num_batches is not None:
        # don't count if it's the entire dataset
        actual_num_batches = 0
        num_total_samples = 0
        for x, _ in dataloader:
            actual_num_batches += 1
            num_total_samples += len(x)

        assert actual_num_batches == subset_num_batches
        assert num_total_samples == batch_size * subset_num_batches // world_size
