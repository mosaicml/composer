# Copyright 2021 MosaicML. All Rights Reserved.

"""
See the :doc:`Method Card</model_cards/imagenet_resnet` for more details.

"""
from composer.models.resnet.model import ComposerResNet as ComposerResNet
from composer.models.resnet.resnet_hparams import ResNetHparams as ResNetHparams

__all__ = [ComposerResNet, ResNetHparams]

_metadata = {
    'resnet18': {
        '_task': 'Image Classification',
        '_dataset': 'ImageNet',
        '_name': 'ResNet18',
        '_quality': 'TBD',
        '_metric': 'Top-1 Accuracy',
        '_ttt': 'TBD',
        '_hparams': 'resnet18.yaml'
    },
    'resnet34': {
        '_task': 'Image Classification',
        '_dataset': 'ImageNet',
        '_name': 'ResNet34',
        '_quality': 'TBD',
        '_metric': 'Top-1 Accuracy',
        '_ttt': 'TBD',
        '_hparams': 'resnet34.yaml'
    },
    'resnet50': {
        '_task': 'Image Classification',
        '_dataset': 'ImageNet',
        '_name': 'ResNet50',
        '_quality': '76.51',
        '_metric': 'Top-1 Accuracy',
        '_ttt': '3h 33m',
        '_hparams': 'resnet50.yaml'
    },
    'resnet101': {
        '_task': 'Image Classification',
        '_dataset': 'ImageNet',
        '_name': 'ResNet101',
        '_quality': '78.10',
        '_metric': 'Top-1 Accuracy',
        '_ttt': '8h 15m',
        '_hparams': 'resnet101.yaml',
    },
    'resnet152': {
        '_task': 'Image Classification',
        '_dataset': 'ImageNet',
        '_name': 'ResNet152',
        '_quality': 'TBD',
        '_metric': 'Top-1 Accuracy',
        '_ttt': 'TBD',
        '_hparams': 'resnet152.yaml'
    }
}
