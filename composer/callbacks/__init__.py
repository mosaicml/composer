# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callbacks that run at each training loop :class:`~composer.core.event.Event`.

Each callback inherits from the :class:`~composer.core.callback.Callback` base class. See detailed description and
examples for writing your own callbacks at the :class:`~composer.core.callback.Callback` base class.
"""
from composer.callbacks.callback import Callback
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.early_stopper import EarlyStopper
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.mlperf import MLPerfCallback
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.callbacks.threshold_stopper import ThresholdStopper

__all__ = [
    "Callback",
    "GradMonitor",
    "LRMonitor",
    "MemoryMonitor",
    "SpeedMonitor",
    "CheckpointSaver",
    "MLPerfCallback",
    "EarlyStopper",
    "ThresholdStopper",
    "callback_registry",
]

callback_registry = {
    "checkpoint_saver": CheckpointSaver,
    "speed_monitor": SpeedMonitor,
    "lr_monitor": LRMonitor,
    "grad_monitor": GradMonitor,
    "memory_monitor": MemoryMonitor,
    "mlperf": MLPerfCallback,
    "early_stopper": EarlyStopper,
    "threshold_stopper": ThresholdStopper,
}
