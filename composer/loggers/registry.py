# Copyright 2021 MosaicML. All Rights Reserved.

"""Logger registry."""

from typing import Type

from composer.loggers.logger_hparams import (FileLoggerHparams, InMemoryLoggerHparams, LoggerDestinationHparams,
                                             ProgressBarLoggerHparams, WandBLoggerHparams)

__all__ = ["get_logger_hparams", "logger_registry"]

logger_registry = {
    "file": FileLoggerHparams,
    "wandb": WandBLoggerHparams,
    "tqdm": ProgressBarLoggerHparams,
    "in_memory": InMemoryLoggerHparams,
}


def get_logger_hparams(name: str) -> Type[LoggerDestinationHparams]:
    """Returns LoggerDestinationHparams class for a given logger type.

    Args:
        name (str): Logger type of the returned hparams object.  One of
                            (``"file"``, ``"wandb"``, ``"tqdm``, ``"in_memory"``).

    Returns:
        Type[LoggerDestinationHparams]: LoggerDestinationHparams of specified type.
    """
    return logger_registry[name]
