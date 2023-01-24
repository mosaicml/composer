# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The models module contains the :class:`.ComposerModel` base class along with reference
implementations of many common models. Additionally, it includes task-specific convenience
:class:`.ComposerModel`\\s that wrap existing Pytorch models with standard forward passes
and logging to enable quick interaction with the :class:`.Trainer`.

See :doc:`Composer Model </composer_model>` for more details.
"""

from composer.models.base import ComposerModel
from composer.models.bert import create_bert_classification, create_bert_mlm
from composer.models.classify_mnist import mnist_model
from composer.models.deeplabv3 import composer_deeplabv3
from composer.models.efficientnetb0 import composer_efficientnetb0
from composer.models.gpt2 import create_gpt2
from composer.models.huggingface import HuggingFaceModel
from composer.models.initializers import Initializer
from composer.models.mmdetection import MMDetModel
from composer.models.resnet import composer_resnet
from composer.models.resnet_cifar import composer_resnet_cifar
from composer.models.tasks import ComposerClassifier
from composer.models.timm import composer_timm
from composer.models.unet import UNet
from composer.models.vit_small_patch16 import vit_small_patch16

__all__ = [
    'ComposerModel',
    'create_bert_classification',
    'create_bert_mlm',
    'mnist_model',
    'composer_deeplabv3',
    'composer_efficientnetb0',
    'create_gpt2',
    'HuggingFaceModel',
    'Initializer',
    'MMDetModel',
    'composer_resnet',
    'composer_resnet_cifar',
    'ComposerClassifier',
    'composer_timm',
    'UNet',
    'vit_small_patch16',
]
