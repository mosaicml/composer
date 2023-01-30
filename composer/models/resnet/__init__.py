# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The ResNet model family is a set of convolutional neural networks described in `Deep Residual Learning for Image
Recognition <https://arxiv.org/abs/1512.03385>`_ (He et al, 2015). ResNets can be used as the base for a variety of
vision tasks. ImageNet ResNets are a subset of the ResNet family which were designed specifically for classification on
the ImageNet dataset.

See the :doc:`Model Card </model_cards/resnet>` for more details.
"""
from composer.models.resnet.model import composer_resnet

__all__ = ['composer_resnet']

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
