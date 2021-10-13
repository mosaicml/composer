# Copyright 2021 MosaicML. All Rights Reserved.

import pytest

from composer.models import BaseMosaicModel, ModelHparams
from composer.trainer.trainer_hparams import model_registry


@pytest.mark.parametrize("model_name", model_registry.keys())
def test_model_registry(model_name, request):
    if model_name == 'gpt2':  # do not pull from HF model hub
        request.applymarker(pytest.mark.xfail())

    # create the model hparams object
    model_hparams = model_registry[model_name]()

    requires_num_classes = set([
        "resnet56_cifar10",
        "efficientnetb0",
        "resnet101",
        "resnet50",
        "resnet18",
        "mnist_classifier",
    ])
    if model_name in requires_num_classes:
        model_hparams.num_classes = 10

    assert isinstance(model_hparams, ModelHparams)

    # create the model object using the hparams
    model = model_hparams.initialize_object()
    assert isinstance(model, BaseMosaicModel)
