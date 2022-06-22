# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Tensorboard <https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more./>`_."""

from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from composer.core.state import State
from composer.loggers.logger import LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist

__all__ = ['TensorboardLogger']


class TensorboardLogger(LoggerDestination):
    """Log to `Tensorboard <https://www.tensorflow.org/tensorboard/>`_.

    Args:
        log_dir (str, optional): The path to the directory to put these logs in. If it is
            None the logs will be placed in `./runs/{month}{day}{HH-MM-SS}_{device_name}.local
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
    """

    def __init__(self, log_dir: Optional[str] = None, rank_zero_only: bool = True):

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.rank_zero_only = rank_zero_only

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        del log_level

        if (not self.rank_zero_only) or dist.get_global_rank() == 0:
            for tag, data_point in data.items():
                if isinstance(data_point, str):
                    self.writer.add_text(tag, data_point)
                self.writer.add_scalar(tag, data_point, global_step=int(state.timestamp.batch))
