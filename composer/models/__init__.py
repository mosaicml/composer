# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The models module contains the :class:`.ComposerModel` base class along with reference
implementations of many common models. Additionally, it includes task-specific convenience
:class:`.ComposerModel`\\s that wrap existing Pytorch models with standard forward passes
and logging to enable quick interaction with the :class:`.Trainer`.

See :doc:`Composer Model </composer_model>` for more details.
"""

from composer.models.base import ComposerModel as ComposerModel
from composer.models.bert import BERTForClassificationHparams as BERTForClassificationHparams
from composer.models.bert import BERTHparams as BERTHparams
from composer.models.bert import create_bert_classification as create_bert_classification
from composer.models.bert import create_bert_mlm as create_bert_mlm
from composer.models.classify_mnist import MnistClassifierHparams as MnistClassifierHparams
from composer.models.classify_mnist import mnist_model as mnist_model
from composer.models.deeplabv3 import DeepLabV3Hparams as DeepLabV3Hparams
from composer.models.deeplabv3 import composer_deeplabv3 as composer_deeplabv3
from composer.models.efficientnetb0 import EfficientNetB0Hparams as EfficientNetB0Hparams
from composer.models.efficientnetb0 import composer_efficientnetb0 as composer_efficientnetb0
from composer.models.gpt2 import GPT2Hparams as GPT2Hparams
from composer.models.gpt2 import create_gpt2 as create_gpt2
from composer.models.huggingface import HuggingFaceModel as HuggingFaceModel
from composer.models.initializers import Initializer as Initializer
from composer.models.model_hparams import ModelHparams as ModelHparams
from composer.models.resnet import ResNetHparams as ResNetHparams
from composer.models.resnet import composer_resnet as composer_resnet
from composer.models.resnet_cifar import ResNetCIFARHparams as ResNetCIFARHparams
from composer.models.resnet_cifar import composer_resnet_cifar as composer_resnet_cifar
from composer.models.ssd import SSD as SSD
from composer.models.ssd import SSDHparams as SSDHparams
from composer.models.tasks import ComposerClassifier as ComposerClassifier
from composer.models.timm import TimmHparams as TimmHparams
from composer.models.timm import composer_timm as composer_timm
from composer.models.unet import UNet as UNet
from composer.models.unet import UnetHparams as UnetHparams
from composer.models.vit_small_patch16 import ViTSmallPatch16Hparams as ViTSmallPatch16Hparams
from composer.models.vit_small_patch16 import vit_small_patch16 as vit_small_patch16
