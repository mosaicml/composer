# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callbacks that run at each training loop :class:`.Event`.

Each callback inherits from the :class:`.Callback` base class. See detailed description and
examples for writing your own callbacks at the :class:`.Callback` base class.
"""
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.early_stopper import EarlyStopper
from composer.callbacks.export_for_inference import ExportForInferenceCallback
from composer.callbacks.image_visualizer import ImageVisualizer
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.mlperf import MLPerfCallback
from composer.callbacks.optimizer_monitor import OptimizerMonitor
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.callbacks.threshold_stopper import ThresholdStopper

__all__ = [
    'OptimizerMonitor',
    'LRMonitor',
    'MemoryMonitor',
    'SpeedMonitor',
    'CheckpointSaver',
    'MLPerfCallback',
    'EarlyStopper',
    'ExportForInferenceCallback',
    'ThresholdStopper',
    'ImageVisualizer',
]
