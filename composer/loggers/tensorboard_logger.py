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

    def __init__(self, log_dir: str = None, run_name: Optional[str] = None, rank_zero_only: bool = True):

        self.run_name = run_name if run_name is not None else ''
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, filename_suffix=self.run_name)
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0
        self.run_name = run_name

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):

        if self._enabled:
            for tag, data_point in data.items():
                if isinstance(data_point, str):
                    self.writer.add_text(tag, data_point)
                self.writer.add_scalar(tag, data_point, global_step=int(state.timestamp.batch))
