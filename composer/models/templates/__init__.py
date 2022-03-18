# Copyright 2021 MosaicML. All Rights Reserved.

"""Model templates are ComposerModels with forward passes and logging built-in for many common deep learning tasks."""

from composer.models.templates.classification import ComposerClassifier as ComposerClassifier

__all__ = ["ComposerClassifier"]
