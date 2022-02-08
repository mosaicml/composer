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
from composer.loggers.file_logger import FileLoggerBackend
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger_hparams import (BaseLoggerBackendHparams, FileLoggerBackendHparams, InMemoryLoggerHaparms,
                                             MosaicMLLoggerBackendHparams, TQDMLoggerBackendHparams,
                                             WandBLoggerBackendHparams)
from composer.loggers.mosaicml_logger import MosaicMLLoggerBackend
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.loggers.wandb_logger import WandBLoggerBackend

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "FileLoggerBackend",
    "InMemoryLogger",
    "BaseLoggerBackendHparams",
    "FileLoggerBackendHparams",
    "InMemoryLoggerHaparms",
    "MosaicMLLoggerBackendHparams",
    "TQDMLoggerBackendHparams",
    "WandBLoggerBackendHparams",
    "MosaicMLLoggerBackend",
    "TQDMLoggerBackend",
    "WandBLoggerBackend",
]
