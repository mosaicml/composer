# TODO move this logic somewhere
from typing import Any
from composer.loggers.file_logger import FileLogger
from composer.loggers.slack_logger import SlackLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.loggers.mlflow_logger import MLFlowLogger
from composer.loggers.neptune_logger import NeptuneLogger
from composer.loggers.console_logger import ConsoleLogger
from composer.loggers.cometml_logger import CometMLLogger
from composer.loggers.mosaicml_logger import MosaicMLLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.tensorboard_logger import TensorboardLogger
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader


LOGGER_TYPES = [
    FileLogger,
    SlackLogger,
    WandBLogger,
    MLFlowLogger,
    NeptuneLogger,
    ConsoleLogger,
    CometMLLogger,
    MosaicMLLogger,
    InMemoryLogger,
    TensorboardLogger,
    ProgressBarLogger,
    RemoteUploaderDownloader,
    LoggerDestination,
]

def get_logger_type(logger: Any) -> str:
    for logger_type in LOGGER_TYPES:
        if isinstance(logger, logger_type):
            return logger_type.__name__
    return 'Custom'

