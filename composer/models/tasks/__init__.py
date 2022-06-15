# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Model tasks are ComposerModels with forward passes and logging built-in for many common deep learning tasks."""

from composer.models.tasks.classification import ComposerClassifier as ComposerClassifier

__all__ = ['ComposerClassifier']
