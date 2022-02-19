# Copyright 2021 MosaicML. All Rights Reserved.

"""Logging.

The trainer includes a :class:`~.logger.Logger`, which routes logging
calls to a logger. Each logger inherits from
:class:`~.base_backend.LoggerCallback`, which inherits from
:class:`~.callback.Callback`.

See the :class:`~.base_backend.LoggerCallback` documentation for an example of how to
define a custom logger and use it when training.
"""

from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger_hparams import (FileLoggerHparams, InMemoryLoggerHaparms, LoggerCallbackHparams,
                                             TQDMLoggerHparams, WandBLoggerHparams)
from composer.loggers.tqdm_logger import TQDMLogger
from composer.loggers.wandb_logger import WandBLogger

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "FileLogger",
    "InMemoryLogger",
    "LoggerCallbackHparams",
    "FileLoggerHparams",
    "InMemoryLoggerHaparms",
    "TQDMLoggerHparams",
    "WandBLoggerHparams",
    "TQDMLogger",
    "WandBLogger",
]
