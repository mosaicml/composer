# Copyright 2021 MosaicML. All Rights Reserved.

"""The models module contains the :class:`.ComposerModel` base class along with reference
implementations of many common models. Additionally, it includes task-specific convenience
:class:`.ComposerModel`\\s that wrap existing Pytorch models with standard forward passes
and logging to enable quick interaction with the :class:`.Trainer`.

See :doc:`Composer Model </composer_model>` for more details.
"""

from composer.models.base import ComposerClassifier as ComposerClassifier
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
from composer.models.model_hparams import Initializer as Initializer
from composer.models.model_hparams import ModelHparams as ModelHparams
from composer.models.resnet import ComposerResNet as ComposerResNet
from composer.models.resnet import ResNetHparams as ResNetHparams
from composer.models.resnet9_cifar10 import CIFAR10_ResNet9 as CIFAR10_ResNet9
from composer.models.resnet9_cifar10 import CIFARResNet9Hparams as CIFARResNet9Hparams
from composer.models.resnet20_cifar10 import CIFAR10_ResNet20 as CIFAR10_ResNet20
from composer.models.resnet20_cifar10 import CIFARResNet20Hparams as CIFARResNet20Hparams
from composer.models.resnet56_cifar10 import CIFAR10_ResNet56 as CIFAR10_ResNet56
from composer.models.resnet56_cifar10 import CIFARResNetHparams as CIFARResNetHparams
from composer.models.resnet.model import ComposerResNet as ComposerResNet
from composer.models.resnet.resnet_hparams import ResNetHparams as ResNetHparams
from composer.models.ssd import SSD as SSD
from composer.models.ssd import SSDHparams as SSDHparams
from composer.models.timm import Timm as Timm
from composer.models.timm import TimmHparams as TimmHparams
from composer.models.transformer_shared import ComposerTransformer as ComposerTransformer
from composer.models.unet import UNet as UNet
from composer.models.unet import UnetHparams as UnetHparams
from composer.models.vit_small_patch16 import ViTSmallPatch16 as ViTSmallPatch16
from composer.models.vit_small_patch16 import ViTSmallPatch16Hparams as ViTSmallPatch16Hparams
