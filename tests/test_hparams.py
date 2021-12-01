# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest

import composer
from composer.datasets.hparams import DatasetHparams
from composer.trainer import TrainerHparams
from composer.trainer.devices import CPUDeviceHparams


def walk_model_yamls():
    yamls = []
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(composer.__file__), "yamls", "models")):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith(".yaml"):
                yamls.append(filepath)
    assert len(yamls) > 0, "there should be at least one yaml!"
    return yamls


def _configure_dataset_for_synthetic(dataset_hparams: DatasetHparams) -> None:
    try:
        synthetic = dataset_hparams.synthetic
    except AttributeError:
        pytest.xfail(f"{dataset_hparams.__class__.__name__} does not support synthetic data")
        raise
    if synthetic is None:
        dataset_hparams.synthetic = dataset_hparams.get_synthetic_hparams_cls()()

        try:
            dataset_hparams.num_total_batches = 1
        except AttributeError:
            pytest.xfail(f"{dataset_hparams.__class__.__name__} does not support num total batches")
            raise


@pytest.mark.parametrize("hparams_file", walk_model_yamls())
class TestHparamsCreate:

    def test_hparams_create(self, hparams_file: str):
        hparams = TrainerHparams.create(hparams_file, cli_args=False)
        assert isinstance(hparams, TrainerHparams)

    def test_trainer_initialize(self, hparams_file: str):
        hparams = TrainerHparams.create(hparams_file, cli_args=False)

        _configure_dataset_for_synthetic(hparams.train_dataset)
        _configure_dataset_for_synthetic(hparams.val_dataset)
        hparams.device = CPUDeviceHparams()

        hparams.initialize_object()
