# Copyright 2021 MosaicML. All Rights Reserved.

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