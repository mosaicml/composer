# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Type

from composer.loggers.logger_hparams import (LoggerCallbackHparams, FileLoggerHparams, TQDMLoggerHparams,
                                             WandBLoggerHparams)

logger_registry = {
    "file": FileLoggerHparams,
    "wandb": WandBLoggerHparams,
    "tqdm": TQDMLoggerHparams,
}


def get_logger_hparams(name: str) -> Type[LoggerCallbackHparams]:
    return logger_registry[name]
