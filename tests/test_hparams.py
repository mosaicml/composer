# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest

import composer
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.models import DeepLabV3Hparams, ModelHparams, TransformerHparams
from composer.trainer import TrainerHparams
from composer.trainer.devices import CPUDeviceHparams
from composer.tests.transformer_utils import generate_dummy_model_config


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


def _configure_model_for_synthetic(model_hparams: ModelHparams):
    if isinstance(model_hparams, TransformerHparams):
        # force a non-pretrained model
        model_hparams.pretrained_model_name = None
        model_hparams_name = type(model_hparams).__name__
        model_hparams.model_config = generate_dummy_model_config(model_hparams_name) 

    if isinstance(model_hparams, DeepLabV3Hparams):
        model_hparams.is_backbone_pretrained = False  # prevent downloading pretrained weights during test
        model_hparams.sync_bn = False  # sync_bn throws an error when run on CPU

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

        _configure_dataset_for_synthetic(hparams.train_dataset)
        _configure_model_for_synthetic(hparams.train_dataset)
        if hparams.val_dataset is not None:
            _configure_dataset_for_synthetic(hparams.val_dataset)
        hparams.device = CPUDeviceHparams()

        hparams.initialize_object()
