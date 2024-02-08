# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The models module contains the :class:`.ComposerModel` base class along with reference
implementations of many common models. Additionally, it includes task-specific convenience
:class:`.ComposerModel`\\s that wrap existing Pytorch models with standard forward passes
and logging to enable quick interaction with the :class:`.Trainer`.

See :doc:`Composer Model </composer_model>` for more details.
"""

from composer.models.base import ComposerModel
from composer.models.huggingface import HuggingFaceModel, write_huggingface_pretrained_from_composer_checkpoint
from composer.models.initializers import Initializer
from composer.models.tasks import ComposerClassifier

__all__ = [
    'ComposerModel',
    'HuggingFaceModel',
    'write_huggingface_pretrained_from_composer_checkpoint',
    'Initializer',
    'ComposerClassifier',
]
