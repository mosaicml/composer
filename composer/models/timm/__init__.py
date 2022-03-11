# Copyright 2021 MosaicML. All Rights Reserved.

"""A wrapper around `timm.create_model() <https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model>`_
used to create :class:`.ComposerClassifier`."""

from composer.models.timm.model import Timm as Timm
from composer.models.timm.timm_hparams import TimmHparams as TimmHparams

__all__ = ["Timm", "TimmHparams"]