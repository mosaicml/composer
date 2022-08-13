# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter registry for callbacks."""

from typing import Dict, Type, Union

import yahp as hp

from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.early_stopper import EarlyStopper
from composer.callbacks.export_for_inference import ExportForInferenceCallback
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.image_visualizer import ImageVisualizer
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.mlperf import MLPerfCallback
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.callbacks.threshold_stopper import ThresholdStopper
from composer.core import Callback

callback_registry: Dict[str, Union[Type[Callback], Type[hp.Hparams]]] = {
    'checkpoint_saver': CheckpointSaver,
    'speed_monitor': SpeedMonitor,
    'lr_monitor': LRMonitor,
    'grad_monitor': GradMonitor,
    'memory_monitor': MemoryMonitor,
    'mlperf': MLPerfCallback,
    'early_stopper': EarlyStopper,
    'export_for_inference': ExportForInferenceCallback,
    'threshold_stopper': ThresholdStopper,
    'image_visualizer': ImageVisualizer,
}
