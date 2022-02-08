# Copyright 2021 MosaicML. All Rights Reserved.

from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger_hparams import (FileLoggerHparams, InMemoryLoggerHaparms, LoggerCallbackHparams,
                                             MosaicMLLoggerHparams, TQDMLoggerHparams, WandBLoggerHparams)
from composer.loggers.mosaicml_logger import MosaicMLLogger
from composer.loggers.tqdm_logger import TQDMLogger
from composer.loggers.wandb_logger import WandBLogger

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "FileLogger",
    "InMemoryLogger",
    "LoggerCallbackHparams",
    "FileLoggerHparams",
    "InMemoryLoggerHaparms",
    "MosaicMLLoggerHparams",
    "TQDMLoggerHparams",
    "WandBLoggerHparams",
    "MosaicMLLogger",
    "TQDMLogger",
    "WandBLogger",
]
