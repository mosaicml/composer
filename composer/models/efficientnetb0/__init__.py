# Copyright 2021 MosaicML. All Rights Reserved.

"""The EfficientNet model family is a set of convolutional neural networks that can be used as the basis for a variety
of vision tasks, but were initially designed for image classification. The model family was designed to reach the
highest accuracy for a given computation budget during inference by simultaneously scaling model depth, model width, and
image resolution according to an empirically determined scaling law.

See the :doc:`Model Card </model_cards/efficientnet>` for more details.
"""
from composer.models.efficientnetb0.efficientnetb0_hparams import EfficientNetB0Hparams as EfficientNetB0Hparams
from composer.models.efficientnetb0.model import EfficientNetB0 as EfficientNetB0

__all__ = ["EfficientNetB0", "EfficientNetB0Hparams"]

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'EfficientNet-B0'
_quality = '76.63'
_metric = 'Top-1 Accuracy'
_ttt = '21h 48m'
_hparams = 'efficientnetb0.yaml'
