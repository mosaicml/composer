# Copyright 2021 MosaicML. All Rights Reserved.

"""Logging.

The trainer includes a :class:`~composer.core.logging.logger.Logger`, which routes logging
calls to a logger. Each logger inherits from
:class:`~composer.core.logging.base_backend.LoggerCallback`, which inherits from
:class:`Callback`.

For example, to define a new logger and use it when training:

.. code-block:: python

    from composer.core.logging import LoggerCallback

    class MyLogger(LoggerCallback)

        def log_metric(self, timestamp, log_level, data):
            print(f'Timestamp: {timestamp}: {log_level} {data}')

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration="1ep",
        optimizers=[optimizer],
        loggers=[MyLogger()]
    )
"""
from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger_hparams import (FileLoggerHparams, InMemoryLoggerHaparms, LoggerCallbackHparams,
                                             TQDMLoggerHparams, WandBLoggerHparams)
from composer.loggers.tqdm_logger import TQDMLogger
from composer.loggers.wandb_logger import WandBLogger

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "FileLogger",
    "InMemoryLogger",
    "LoggerCallbackHparams",
    "FileLoggerHparams",
    "InMemoryLoggerHaparms",
    "TQDMLoggerHparams",
    "WandBLoggerHparams",
    "TQDMLogger",
    "WandBLogger",
]
