# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper around `timm.create_model() <https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model>`_
used to create :class:`.ComposerClassifier`."""

from composer.models.timm.model import composer_timm as composer_timm

__all__ = ['composer_timm']
