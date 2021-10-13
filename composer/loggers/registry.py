# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Type

from composer.loggers.logger_hparams import (BaseLoggerBackendHparams, FileLoggerBackendHparams,
                                             TQDMLoggerBackendHparams, WandBLoggerBackendHparams)

logger_registry = {
    "file": FileLoggerBackendHparams,
    "wandb": WandBLoggerBackendHparams,
    "tqdm": TQDMLoggerBackendHparams,
}


def get_logger_hparams(name: str) -> Type[BaseLoggerBackendHparams]:
    return logger_registry[name]
