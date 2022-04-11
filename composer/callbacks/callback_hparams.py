# Copyright 2021 MosaicML. All Rights Reserved.

"""Hyperparameters for callbacks."""
from __future__ import annotations

import abc
import textwrap
from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.core.callback import Callback
from composer.core.time import Time
from composer.utils import import_object

__all__ = [
    "CallbackHparams",
    "GradMonitorHparams",
    "MemoryMonitorHparams",
    "LRMonitorHparams",
    "SpeedMonitorHparams",
    "CheckpointSaverHparams",
]


@dataclass
class CallbackHparams(hp.Hparams, abc.ABC):
    """Base class for Callback hyperparameters."""

    @abc.abstractmethod
    def initialize_object(self) -> Callback:
        """Initialize the callback.

        Returns:
            Callback: An instance of the callback.
        """
        pass


@dataclass
class GradMonitorHparams(CallbackHparams):
    """:class:`~.GradMonitor` hyperparamters.

    Args:
        log_layer_grad_norms (bool, optional): 
            See :class:`~.GradMonitor` for documentation.
    """

    log_layer_grad_norms: bool = hp.optional(
        doc="Whether to log gradient norms for individual layers.",
        default=False,
    )

    def initialize_object(self) -> GradMonitor:
        """Initialize the GradMonitor callback.

        Returns:
            GradMonitor: An instance of :class:`~.GradMonitor`.
        """
        return GradMonitor(log_layer_grad_norms=self.log_layer_grad_norms)


@dataclass
class MemoryMonitorHparams(CallbackHparams):
    """:class:`~.MemoryMonitor` hyperparameters.

    There are no parameters as :class:`~.MemoryMonitor` does not take any parameters.
    """

    def initialize_object(self) -> MemoryMonitor:
        """Initialize the MemoryMonitor callback.

        Returns:
            MemoryMonitor: An instance of :class:`~.MemoryMonitor`.
        """
        return MemoryMonitor()


@dataclass
class LRMonitorHparams(CallbackHparams):
    """:class:`~.LRMonitor` hyperparameters.

    There are no parameters as :class:`~.LRMonitor` does not take any parameters.
    """

    def initialize_object(self) -> LRMonitor:
        """Initialize the LRMonitor callback.

        Returns:
            LRMonitor: An instance of :class:`~.LRMonitor`.
        """
        return LRMonitor()


@dataclass
class SpeedMonitorHparams(CallbackHparams):
    """:class:`~.SpeedMonitor` hyperparameters.

    Args:
        window_size (int, optional): See :class:`~.SpeedMonitor` for documentation.
    """
    window_size: int = hp.optional(
        doc="Number of batchs to use for a rolling average of throughput.",
        default=100,
    )

    def initialize_object(self) -> SpeedMonitor:
        """Initialize the SpeedMonitor callback.

        Returns:
            SpeedMonitor: An instance of :class:`~.SpeedMonitor`.
        """
        return SpeedMonitor(window_size=self.window_size)


@dataclass
class CheckpointSaverHparams(CallbackHparams):
    """:class:`~.CheckpointSaver` hyperparameters.
    
    Args:
        save_folder (str, optional): See :class:`~.CheckpointSaver`.
        filename (str, optional): See :class:`~.CheckpointSaver`.
        artifact_name (str, optional): See :class:`~.CheckpointSaver`.
        latest_filename (str, optional): See :class:`~.CheckpointSaver`.
        overwrite (str, optional): See :class:`~.CheckpointSaver`.
        weights_only (bool, optional): See :class:`~.CheckpointSaver`.
        num_checkpoints_to_keep (int, optional): See :class:`~.CheckpointSaver`.

        save_interval (str, optional): Either a :doc:`time-string </trainer/time>` or a path to a function.

            If a :doc:`time-string </trainer/time>`, checkpoints will be saved according to this interval.

            If a path to a function, it should be of the format ``'path.to.function:function_name'``. The function
            should take (:class:`~.State`, :class:`~.Event`) and return a
            boolean indicating whether a checkpoint should be saved given the current state and event. The event will
            be either :attr:`~composer.core.event.Event.BATCH_CHECKPOINT` or
            :attr:`~composer.core.event.Event.EPOCH_CHECKPOINT`.
    """
    save_folder: str = hp.optional(doc="Folder where checkpoints will be saved.", default="{run_name}/checkpoints")
    filename: str = hp.optional("Checkpoint name format string.", default="ep{epoch}-ba{batch}-rank{rank}")
    artifact_name: str = hp.optional("Checkpoint artifact name format string.",
                                     default="{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}")
    latest_filename: Optional[str] = hp.optional("Latest checkpoint symlink format string.",
                                                 default="latest-rank{rank}")
    overwrite: bool = hp.optional("Whether to override existing checkpoints.", default=False)
    weights_only: bool = hp.optional("Whether to save only checkpoint weights", default=False)
    save_interval: str = hp.optional(textwrap.dedent("""\
        Checkpoint interval or path to a `(State, Event) -> bool` function
        returning whether a checkpoint should be saved."""),
                                     default="1ep")
    num_checkpoints_to_keep: int = hp.optional(
        "Number of checkpoints to persist locally. Set to -1 to never delete checkpoints.",
        default=-1,
    )

    def initialize_object(self) -> CheckpointSaver:
        try:
            save_interval = Time.from_timestring(self.save_interval)
        except ValueError:
            # assume it is a function path
            save_interval = import_object(self.save_interval)
        return CheckpointSaver(
            folder=self.save_folder,
            filename=self.filename,
            artifact_name=self.artifact_name,
            latest_filename=self.latest_filename,
            overwrite=self.overwrite,
            save_interval=save_interval,
            weights_only=self.weights_only,
            num_checkpoints_to_keep=self.num_checkpoints_to_keep,
        )
