# Copyright 2021 MosaicML. All Rights Reserved.

"""Logging.

The trainer includes a :class:`~composer.core.logging.Logger`, which routes logging calls to logger backends.
Each logger backend inherits from :class:`~composer.core.logging.base_backend.BaseLoggerBackend`,
which inherits from :class:`Callback`.

For example, to define a new logging backend:

.. code-block:: python

    from composer.core.logging import BaseLoggerBackend

    class MyLoggerBackend(BaseLoggerBackend)

        def log_metric(self, epoch, step, log_level, data):
            print(f'Epoch {epoch} Step {step}: {log_level} {data}')
"""
from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger_hparams import (FileLoggerHparams, InMemoryLoggerHparams, LoggerCallbackHparams,
                                             TQDMLoggerHparams, WandBLoggerHparams)
from composer.loggers.tqdm_logger import TQDMLogger
from composer.loggers.wandb_logger import WandBLogger

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "FileLogger",
    "InMemoryLogger",
    "LoggerCallbackHparams",
    "FileLoggerHparams",
    "InMemoryLoggerHparams",
    "TQDMLoggerHparams",
    "WandBLoggerHparams",
    "TQDMLogger",
    "WandBLogger",
]
