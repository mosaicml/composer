# Copyright 2021 MosaicML. All Rights Reserved.

"""Hyperparameters for callbacks."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yahp as hp

from composer.core.callback import Callback

if TYPE_CHECKING:
    from composer.callbacks.grad_monitor import GradMonitor
    from composer.callbacks.lr_monitor import LRMonitor
    from composer.callbacks.memory_monitor import MemoryMonitor
    from composer.callbacks.speed_monitor import SpeedMonitor

__all__ = [
    "CallbackHparams",
    "GradMonitorHparams",
    "MemoryMonitorHparams",
    "LRMonitorHparams",
    "SpeedMonitorHparams",
]


@dataclass
class CallbackHparams(hp.Hparams, abc.ABC):
    """Base class for callback hyperparameters.

    Callback parameters that are added to the callbacks argument of
    :attr:`~composer.trainer.trainer_hparams.TrainerHparams` (e.g., via YAML or the CLI). See `Trainer with YAHP <https://docs.mosaicml.com/en/latest/tutorials/adding_models_datasets.html#trainer-with-yahp>`_ for more details.
    These are initialized in the training loop.
    """

    @abc.abstractmethod
    def initialize_object(self) -> Callback:
        """Initialize the callback.

        Returns:
            Callback: An instance of the callback.
        """
        pass


@dataclass
class GradMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.grad_monitor.GradMonitor` hyperparamters.

    Args:
        log_layer_grad_norms (bool, optional): 
            See :class:`~composer.callbacks.grad_monitor.GradMonitor` for documentation.
    """

    log_layer_grad_norms: bool = hp.optional(
        doc="Whether to log gradient norms for individual layers.",
        default=False,
    )

    def initialize_object(self) -> GradMonitor:
        """Initialize the GradMonitor callback.

        Returns:
            GradMonitor: An instance of :mod:`~composer.callbacks.grad_monitor.GradMonitor`.
        """
        from composer.callbacks.grad_monitor import GradMonitor
        return GradMonitor(log_layer_grad_norms=self.log_layer_grad_norms)


@dataclass
class MemoryMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.memory_monitor.MemoryMonitor` hyperparameters.

    There are no parameters as :class:`~composer.callbacks.memory_monitor.MemoryMonitor` does not take any parameters.
    """

    def initialize_object(self) -> MemoryMonitor:
        """Initialize the MemoryMonitor callback.

        Returns:
            MemoryMonitor: An instance of :mod:`~composer.callbacks.memory_monitor.MemoryMonitor`.
        """
        from composer.callbacks.memory_monitor import MemoryMonitor
        return MemoryMonitor()


@dataclass
class LRMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.lr_monitor.LRMonitor` hyperparameters.

    There are no parameters as :class:`~composer.callbacks.lr_monitor.LRMonitor` does not take any parameters.
    """

    def initialize_object(self) -> LRMonitor:
        """Initialize the LRMonitor callback.

        Returns:
            LRMonitor: An instance of :mod:`~composer.callbacks.lr_monitor.LRMonitor`.
        """
        from composer.callbacks.lr_monitor import LRMonitor
        return LRMonitor()


@dataclass
class SpeedMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.speed_monitor.SpeedMonitor` hyperparameters.

    Args:
        window_size (int, optional):
            See :class:`~composer.callbacks.speed_monitor.SpeedMonitor` for documentation.
    """
    window_size: int = hp.optional(
        doc="Number of batchs to use for a rolling average of throughput.",
        default=100,
    )

    def initialize_object(self) -> SpeedMonitor:
        """Initialize the SpeedMonitor callback.

        Returns:
            SpeedMonitor: An instance of :mod:`~composer.callbacks.speed_monitor.SpeedMonitor`.
        """
        from composer.callbacks.speed_monitor import SpeedMonitor
        return SpeedMonitor(window_size=self.window_size)
