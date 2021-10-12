# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
from dataclasses import dataclass

import yahp as hp

from composer.core.callback import Callback


@dataclass
class CallbackHparams(hp.Hparams, abc.ABC):
    """Base class for :class:`~composer.core.callback.Callback` hyperparameters.
    
    Callback parameters that are added to
    :attr:`composer.trainer.trainer_hparams.TrainerHparams.callbacks`
    (e.g. via YAML or the CLI) are initialized in the training loop.
    """

    @abc.abstractmethod
    def initialize_object(self) -> Callback:
        """Initialize the callback.

        Returns:
            Callback: An instance of the callback.
        """
        pass
