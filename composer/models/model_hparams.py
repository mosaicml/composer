# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""General `YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for Base ComposerModels."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import yahp as hp

from composer.models.base import ComposerModel
from composer.models.initializers import Initializer

__all__ = ['ModelHparams']


@dataclass
class ModelHparams(hp.Hparams, ABC):
    """General `YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for ComposerModels.

    Args:
        num_classes (int): The number of classes. Needed for classification tasks. Default: ``None``.
        initializers (List[Initializer], optional): The initialization strategy for the model. Default: ``None``.
    """
    initializers: List[Initializer] = hp.optional(
        default_factory=lambda: [],
        doc='The initialization strategy for the model',
    )

    num_classes: Optional[int] = hp.optional(
        doc='The number of classes.  Needed for classification tasks',
        default=None,
    )

    @abstractmethod
    def initialize_object(self) -> ComposerModel:
        """
        Construct a :class:`.ComposerModel`.

        Returns:
            ComposerModel: The constructed :class:`.ComposerModel`
        """
        pass
