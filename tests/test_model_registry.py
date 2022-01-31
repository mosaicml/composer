# Copyright 2021 MosaicML. All Rights Reserved.

import pytest

from composer.models import ModelHparams
from composer.trainer.trainer_hparams import model_registry


@pytest.mark.parametrize("model_name", model_registry.keys())
def test_model_registry(model_name, request):
    if model_name in ['timm']:
        pytest.importorskip("timm")
    if model_name in ['unet']:
        pytest.importorskip("monai")

    # TODO (Moin + Ravi): create dummy versions of these models to pass unit tests
    if model_name in ['gpt2', 'bert', 'bert_classification']:  # do not pull from HF model hub
        request.applymarker(pytest.mark.xfail())

    # create the model hparams object
    model_hparams = model_registry[model_name]()

    requires_num_classes = set([
        "deeplabv3",
        "resnet9_cifar10",
        "resnet56_cifar10",
        "efficientnetb0",
        "resnet101",
        "resnet50",
        "resnet18",
        "mnist_classifier",
    ])
    if model_name in requires_num_classes:
        model_hparams.num_classes = 10

    if model_name == "deeplabv3":
        model_hparams.is_backbone_pretrained = False

    if model_name == "timm":
        model_hparams.model_name = "resnet18"

    assert isinstance(model_hparams, ModelHparams)
