# Copyright 2021 MosaicML. All Rights Reserved.

"""Logger registry."""

from typing import Type

from composer.loggers.logger_hparams import (FileLoggerHparams, InMemoryLoggerHaparms, LoggerCallbackHparams,
                                             TQDMLoggerHparams, WandBLoggerHparams)

__all__ = ["get_logger_hparams", "logger_registry"]

logger_registry = {
    "file": FileLoggerHparams,
    "wandb": WandBLoggerHparams,
    "tqdm": TQDMLoggerHparams,
    "in_memory": InMemoryLoggerHaparms,
}


def get_logger_hparams(name: str) -> Type[LoggerCallbackHparams]:
    """Returns LoggerCallbackHparams class for a given logger type.

    Args:
        name (str): Logger type to return hparams object of. One of {``"file"``,
        ``"wandb"``, ``"tqdm``, ``"in_memory"``}.

    Returns:
        Type[LoggerCallbackHparams]: LoggerCallbackHparams of specified type.
    """
    return logger_registry[name]
