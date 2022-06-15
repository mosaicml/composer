# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.models import ModelHparams
from composer.trainer.trainer_hparams import model_registry


@pytest.mark.parametrize('model_name', model_registry.keys())
def test_model_registry(model_name, request):
    if model_name in ['timm']:
        pytest.importorskip('timm')
    if model_name in ['unet']:
        pytest.importorskip('monai')

    # create the model hparams object
    model_hparams = model_registry[model_name]()

    requires_num_classes = set([
        'deeplabv3',
        'resnet_cifar',
        'efficientnetb0',
        'resnet',
        'mnist_classifier',
    ])
    if model_name in requires_num_classes:
        model_hparams.num_classes = 10

    if model_name == 'resnet':
        model_hparams.model_name = 'resnet50'

    if model_name == 'deeplabv3':
        model_hparams.is_backbone_pretrained = False

    if model_name == 'timm':
        model_hparams.model_name = 'resnet18'

    assert isinstance(model_hparams, ModelHparams)
