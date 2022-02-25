# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest

import composer
from composer.trainer import TrainerHparams
from composer.trainer.devices import CPUDeviceHparams
from tests.utils.synthetic_utils import configure_dataset_for_synthetic, configure_model_for_synthetic


def walk_model_yamls():
    yamls = []
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(composer.__file__), "yamls", "models")):
        del dirs  # unused
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith(".yaml"):
                yamls.append(filepath)
    assert len(yamls) > 0, "there should be at least one yaml!"
    return yamls


@pytest.mark.timeout(10)
@pytest.mark.parametrize("hparams_file", walk_model_yamls())
class TestHparamsCreate:

    def test_hparams_create(self, hparams_file: str):
        if "timm" in hparams_file:
            pytest.importorskip("timm")
        if hparams_file in ["unet.yaml"]:
            pytest.importorskip("monai")

        hparams = TrainerHparams.create(hparams_file, cli_args=False)
        assert isinstance(hparams, TrainerHparams)

    def test_trainer_initialize(self, hparams_file: str):
        if "timm" in hparams_file:
            pytest.importorskip("timm")
        if hparams_file in ["unet.yaml"]:
            pytest.importorskip("monai")
        hparams = TrainerHparams.create(hparams_file, cli_args=False)
        hparams.dataloader.num_workers = 0
        hparams.dataloader.persistent_workers = False
        hparams.dataloader.pin_memory = False
        hparams.dataloader.prefetch_factor = 2

        configure_dataset_for_synthetic(hparams.train_dataset)
        configure_model_for_synthetic(hparams.model)
        if hparams.val_dataset is not None:
            configure_dataset_for_synthetic(hparams.val_dataset)
        hparams.device = CPUDeviceHparams()
        hparams.load_path = None

        hparams.initialize_object()
