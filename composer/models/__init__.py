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
from composer.models.bert import BERTModel as BERTModel
from composer.models.classify_mnist import MNIST_Classifier as MNIST_Classifier
from composer.models.classify_mnist import MnistClassifierHparams as MnistClassifierHparams
from composer.models.deeplabv3 import ComposerDeepLabV3 as ComposerDeepLabV3
from composer.models.deeplabv3 import DeepLabV3Hparams as DeepLabV3Hparams
from composer.models.efficientnetb0 import EfficientNetB0 as EfficientNetB0
from composer.models.efficientnetb0 import EfficientNetB0Hparams as EfficientNetB0Hparams
from composer.models.gpt2 import GPT2Hparams as GPT2Hparams
from composer.models.gpt2 import GPT2Model as GPT2Model
from composer.models.initializers import Initializer as Initializer
from composer.models.model_hparams import ModelHparams as ModelHparams
from composer.models.resnet import ComposerResNet as ComposerResNet
from composer.models.resnet import ResNetHparams as ResNetHparams
from composer.models.resnet_cifar import ComposerResNetCIFAR as ComposerResNetCIFAR
from composer.models.resnet_cifar import ResNetCIFARHparams as ResNetCIFARHparams
from composer.models.ssd import SSD as SSD
from composer.models.ssd import SSDHparams as SSDHparams
from composer.models.tasks import ComposerClassifier as ComposerClassifier
from composer.models.timm import Timm as Timm
from composer.models.timm import TimmHparams as TimmHparams
from composer.models.transformer_hparams import TransformerHparams as TransformerHparams
from composer.models.transformer_shared import ComposerTransformer as ComposerTransformer
from composer.models.unet import UNet as UNet
from composer.models.unet import UnetHparams as UnetHparams
from composer.models.vit_small_patch16 import ViTSmallPatch16 as ViTSmallPatch16
from composer.models.vit_small_patch16 import ViTSmallPatch16Hparams as ViTSmallPatch16Hparams
