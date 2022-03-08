# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest

import composer
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.models import DeepLabV3Hparams
from composer.trainer import TrainerHparams
from composer.trainer.devices import CPUDeviceHparams


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


def _configure_dataset_for_synthetic(dataset_hparams: DatasetHparams) -> None:
    if not isinstance(dataset_hparams, SyntheticHparamsMixin):
        pytest.xfail(f"{dataset_hparams.__class__.__name__} does not support synthetic data or num_total_batches")

    assert isinstance(dataset_hparams, SyntheticHparamsMixin)

    dataset_hparams.use_synthetic = True


@pytest.mark.parametrize("hparams_file", walk_model_yamls())
class TestHparamsCreate:

    def test_hparams_create(self, hparams_file: str):
        if "timm" in hparams_file:
            pytest.importorskip("timm")
        if "vit" in hparams_file:
            pytest.importorskip("vit_pytorch")
        if hparams_file in ["unet.yaml"]:
            pytest.importorskip("monai")

        hparams = TrainerHparams.create(hparams_file, cli_args=False)
        assert isinstance(hparams, TrainerHparams)

    def test_trainer_initialize(self, hparams_file: str):
        if "timm" in hparams_file:
            pytest.importorskip("timm")
        if "vit" in hparams_file:
            pytest.importorskip("vit_pytorch")
        if hparams_file in ["unet.yaml"]:
            pytest.importorskip("monai")

        hparams = TrainerHparams.create(hparams_file, cli_args=False)
        hparams.dataloader.num_workers = 0
        hparams.dataloader.persistent_workers = False
        hparams.dataloader.pin_memory = False
        hparams.dataloader.prefetch_factor = 2

        _configure_dataset_for_synthetic(hparams.train_dataset)
        if hparams.val_dataset is not None:
            _configure_dataset_for_synthetic(hparams.val_dataset)
        hparams.device = CPUDeviceHparams()

        if isinstance(hparams.model, DeepLabV3Hparams):
            hparams.model.is_backbone_pretrained = False  # prevent downloading pretrained weights during test
            hparams.model.sync_bn = False  # sync_bn throws an error when run on CPU

        hparams.initialize_object()
