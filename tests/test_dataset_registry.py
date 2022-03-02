# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Callable, Dict, Type

import pytest

from composer.datasets import (ADE20kDatasetHparams, BratsDatasetHparams, C4DatasetHparams, CIFAR10DatasetHparams,
                               DataloaderHparams, DatasetHparams, GLUEHparams, ImagenetDatasetHparams, LMDatasetHparams,
                               MNISTDatasetHparams, SyntheticHparamsMixin)
from composer.trainer.trainer_hparams import dataset_registry

# for testing, we provide values for required hparams fields
# to initialize test hparams objects
default_required_fields: Dict[Type[DatasetHparams], Callable[[], DatasetHparams]] = {
    #  hparams with empty dicts have no required fields
    CIFAR10DatasetHparams:
        lambda: CIFAR10DatasetHparams(
            is_train=False,
            download=False,
        ),
    ADE20kDatasetHparams:
        lambda: ADE20kDatasetHparams(is_train=False),
    BratsDatasetHparams:
        lambda: BratsDatasetHparams(is_train=False,),
    ImagenetDatasetHparams:
        lambda: ImagenetDatasetHparams(
            is_train=False,
            crop_size=224,
            resize_size=-1,
        ),
    MNISTDatasetHparams:
        lambda: MNISTDatasetHparams(
            is_train=False,
            download=False,
        ),
    LMDatasetHparams:
        lambda: LMDatasetHparams(
            datadir=["hello"],
            split='train',
            tokenizer_name='gpt2',
        ),
    GLUEHparams:
        lambda: GLUEHparams(
            task="rte",
            tokenizer_name="bert-base-uncased",
            split="train",
        ),
    C4DatasetHparams:
        lambda: C4DatasetHparams(
            split="train",
            max_samples=1000,
            max_seq_len=100,
            tokenizer_name="gpt2",
            group_method="concat",
        ),
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
