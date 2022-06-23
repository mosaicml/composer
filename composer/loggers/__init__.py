# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Loggers to store metrics and artifacts.

In Composer, algorithms and callbacks can make calls to the :class:`~.logger.Logger`,
which then routes the calls to the appropriate :class:`~.logger_destination.LoggerDestination` instances.
The :class:`~.logger_destination.LoggerDestination` does the actual logging, for example to a file,
or Weights and Biases.

See the :class:`~.logger_destination.LoggerDestination` documentation for an example of how to
define a custom logger and use it when training.
"""

from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.loggers.tensorboard_logger import TensorboardLogger
from composer.loggers.wandb_logger import WandBLogger

# All needs to be defined properly for sphinx autosummary
__all__ = [
    'Logger',
    'LoggerDestination',
    'LogLevel',
    'FileLogger',
    'InMemoryLogger',
    'ProgressBarLogger',
    'WandBLogger',
    'ObjectStoreLogger',
    'TensorboardLogger',
]
