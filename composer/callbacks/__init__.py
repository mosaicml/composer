# Copyright 2021 MosaicML. All Rights Reserved.

"""Callbacks that run at each training loop :class:`~composer.core.event.Event`.

Each callback inherits from the :class:`~composer.core.callback.Callback` base class. See detailed description and
examples for writing your own callbacks at the :class:`~composer.core.callback.Callback` base class.
"""
from composer.callbacks.callback_hparams import (CallbackHparams, CheckpointSaverHparams, GradMonitorHparams,
                                                 LRMonitorHparams, MemoryMonitorHparams, RunDirectoryUploaderHparams,
                                                 SpeedMonitorHparams)
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.run_directory_uploader import RunDirectoryUploader
from composer.callbacks.speed_monitor import SpeedMonitor

__all__ = [
    "GradMonitor",
    "LRMonitor",
    "MemoryMonitor",
    "RunDirectoryUploader",
    "SpeedMonitor",
    "CheckpointSaver",
    # hparams objects
    "CallbackHparams",
    "CheckpointSaverHparams",
    "GradMonitorHparams",
    "LRMonitorHparams",
    "MemoryMonitorHparams",
    "RunDirectoryUploaderHparams",
    "SpeedMonitorHparams",
]
