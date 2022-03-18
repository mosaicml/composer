# Copyright 2021 MosaicML. All Rights Reserved.

"""A ResNet model family adapted for CIFAR10 image sizes.

See the :doc:`Model Card </model_cards/cifar_resnet>` for more details.
"""

from composer.models.resnet_cifar10.model import CIFAR10ResNet as CIFAR10ResNet
from composer.models.resnet_cifar10.resnet_cifar10_hparams import CIFAR10ResNetHparams as CIFAR10ResNetHparams

__all__ = ["CIFAR10ResNet", "CIFAR10ResNetHparams"]

_metadata = {
    'cifar_resnet9': {
        '_task': 'Image Classification',
        '_dataset': 'CIFAR10',
        '_name': 'ResNet9',
        '_quality': 'tbd',
        '_metric': 'Top-1 Accuracy',
        '_ttt': 'tbd',
        '_hparams': 'resnet9_cifar10.yaml'
    },
    'cifar_resnet20': {
        '_task': 'Image Classification',
        '_dataset': 'CIFAR10',
        '_name': 'ResNet20',
        '_quality': 'tbd',
        '_metric': 'Top-1 Accuracy',
        '_ttt': 'tbd',
        '_hparams': 'resnet20_cifar10.yaml'
    },
    'cifar_resnet56': {
        '_task': 'Image Classification',
        '_dataset': 'CIFAR10',
        '_name': 'ResNet56',
        '_quality': '93.1',
        '_metric': 'Top-1 Accuracy',
        '_ttt': '35m',
        '_hparams': 'resnet56_cifar10.yaml'
    }
}
