# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The EfficientNet model family is a set of convolutional neural networks that can be used as the basis for a variety
of vision tasks, but were initially designed for image classification. The model family was designed to reach the
highest accuracy for a given computation budget during inference by simultaneously scaling model depth, model width, and
image resolution according to an empirically determined scaling law.

See the :doc:`Model Card </model_cards/efficientnet>` for more details.
"""
from composer.models.efficientnetb0.model import composer_efficientnetb0 as composer_efficientnetb0

__all__ = ['composer_efficientnetb0']

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'EfficientNet-B0'
_quality = '76.63'
_metric = 'Top-1 Accuracy'
_ttt = '21h 48m'
_hparams = 'efficientnetb0.yaml'
