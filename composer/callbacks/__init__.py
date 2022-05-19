# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callbacks that run at each training loop :class:`~composer.core.event.Event`.

Each callback inherits from the :class:`~composer.core.callback.Callback` base class. See detailed description and
examples for writing your own callbacks at the :class:`~composer.core.callback.Callback` base class.
"""
from composer.callbacks.callback_hparams import (BenchmarkerHparams, CallbackHparams, CheckpointSaverHparams, EarlyStopperHparams,
                                                 GradMonitorHparams, LRMonitorHparams, MemoryMonitorHparams,
                                                 MLPerfCallbackHparams, SpeedMonitorHparams, ThresholdStopperHparams)
from composer.callbacks.benchmarker import Benchmarker
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.early_stopper import EarlyStopper
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.mlperf import MLPerfCallback
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.callbacks.threshold_stopper import ThresholdStopper

__all__ = [
    "Benchmarker",
    "GradMonitor",
    "LRMonitor",
    "MemoryMonitor",
    "SpeedMonitor",
    "CheckpointSaver",
    "MLPerfCallback",
    "EarlyStopper",
    "ThresholdStopper",
    # hparams objects
    "CallbackHparams",
    "CheckpointSaverHparams",
    "EarlyStopperHparams",
    "GradMonitorHparams",
    "LRMonitorHparams",
    "MemoryMonitorHparams",
    "SpeedMonitorHparams",
    "MLPerfCallbackHparams",
    "EarlyStopperHparams",
    "ThresholdStopperHparams",
]
