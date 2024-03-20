# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for logging analytics with the MosaicMLLogger."""

from typing import Any

from composer.loggers.cometml_logger import CometMLLogger
from composer.loggers.console_logger import ConsoleLogger
from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.mlflow_logger import MLFlowLogger
from composer.loggers.neptune_logger import NeptuneLogger
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.loggers.slack_logger import SlackLogger
from composer.loggers.tensorboard_logger import TensorboardLogger
from composer.loggers.wandb_logger import WandBLogger

LOGGER_TYPES = [
    FileLogger,
    SlackLogger,
    WandBLogger,
    MLFlowLogger,
    NeptuneLogger,
    ConsoleLogger,
    CometMLLogger,
    InMemoryLogger,
    TensorboardLogger,
    ProgressBarLogger,
    RemoteUploaderDownloader,
    LoggerDestination,
]


def get_logger_type(logger: Any) -> str:
    """Returns the type of a logger as a string. If the logger is not a known type, returns 'Custom'."""
    for logger_type in LOGGER_TYPES:
        if isinstance(logger, logger_type):
            return logger_type.__name__
    return 'Other'
