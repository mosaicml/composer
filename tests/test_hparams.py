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


@pytest.mark.timeout(40)
@pytest.mark.parametrize("hparams_file", walk_model_yamls())
class TestHparamsCreate:

    def test_hparams_create(self, hparams_file: str):
        if "timm" in hparams_file:
            pytest.importorskip("timm")
        if "vit" in hparams_file:
            pytest.importorskip("vit_pytorch")
        if hparams_file in ["unet.yaml"]:
            pytest.importorskip("monai")
        if "deeplabv3" in hparams_file:
            pytest.importorskip("mmseg")
        hparams = TrainerHparams.create(hparams_file, cli_args=False)
        assert isinstance(hparams, TrainerHparams)

    def test_trainer_initialize(self, hparams_file: str):
        if "timm" in hparams_file:
            pytest.importorskip("timm")
        if "vit" in hparams_file:
            pytest.importorskip("vit_pytorch")
        if hparams_file in ["unet.yaml"]:
            pytest.importorskip("monai")

        nlp_hparam_keys = ['glue', 'gpt', 'bert']
        # skip tests that require the NLP stack
        if any([i in hparams_file for i in nlp_hparam_keys]):
            pytest.importorskip("transformers")
            pytest.importorskip("datasets")
            pytest.importorskip("tokenizers")

        if "deeplabv3" in hparams_file:
            pytest.importorskip("mmseg")

        hparams = TrainerHparams.create(hparams_file, cli_args=False)
        hparams.dataloader.num_workers = 0
        hparams.dataloader.persistent_workers = False
        hparams.dataloader.pin_memory = False
        hparams.dataloader.prefetch_factor = 2

        configure_dataset_for_synthetic(hparams.train_dataset, model_hparams=hparams.model)
        configure_model_for_synthetic(hparams.model)
        if hparams.val_dataset is not None:
            configure_dataset_for_synthetic(hparams.val_dataset)
        if hparams.evaluators is not None:
            for evaluator in hparams.evaluators:
                configure_dataset_for_synthetic(evaluator.eval_dataset)
        hparams.device = CPUDeviceHparams()
        hparams.load_path = None

        hparams.initialize_object()
