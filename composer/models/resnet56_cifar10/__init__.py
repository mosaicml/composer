# Copyright 2021 MosaicML. All Rights Reserved.

"""The ResNet model family is a set of convolutional neural networks that can be used as the basis for a variety of
vision tasks. CIFAR ResNet models are a subset of this family designed specifically for the CIFAR-10 and CIFAR-100
datasets.

See the :doc:`Method Card </model_cards/cifar_resnet>` for more details.
"""
from composer.models.resnet56_cifar10.model import CIFAR10_ResNet56 as CIFAR10_ResNet56
from composer.models.resnet56_cifar10.resnet56_cifar10_hparams import CIFARResNetHparams as CIFARResNetHparams

_task = 'Image Classification'
_dataset = 'CIFAR10'
_name = 'ResNet56'
_quality = '93.1'
_metric = 'Top-1 Accuracy'
_ttt = 'tbd'
_hparams = 'resnet56_cifar10.yaml'

__all__ = ["CIFAR10_ResNet56", "CIFARResNetHparams"]
