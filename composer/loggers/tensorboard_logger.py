# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Tensorboard <https://www.tensorflow.org/tensorboard/>`_."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist

__all__ = ['TensorboardLogger']


class TensorboardLogger(LoggerDestination):
    """Log to `Tensorboard <https://www.tensorflow.org/tensorboard/>`_.

    Args:
        log_dir (str, optional): The directory to store the logs. If
            ``None``, the logs will be placed in ``./runs/{month}{day}{HH-MM-SS}_{device_name}.local``.
        artifact_name (str, optional): Format string for the logfile's artifact name.

            The logfile will be periodically logged (according to the ``flush_interval``) as a file artifact.
            The artifact name will be determined by this format string.

            .. seealso:: :meth:`~.log_file_artifact` for file artifact logging.

            The same format variables for ``filename`` are available. Setting this parameter to ``None``
            (the default) will use the same format string as ``filename``. It is sometimes helpful to deviate
            from this default. For example, when ``filename`` contains an absolute path, it is recommended to
            set this parameter explicitly, so the absolute path does not appear in any artifact stores.

            Leading slashes (``'/'``) will be stripped.

            Default: ``None`` (which uses the same format string as ``filename``)
        flush_interval (int, optional): How frequently to flush the log to the file,
            relative to the ``log_level``. For example, if the ``log_level`` is
            :attr:`~.LogLevel.EPOCH`, then the logfile will be flushed every n epochs.  If
            the ``log_level`` is :attr:`~.LogLevel.BATCH`, then the logfile will be
            flushed every n batches. Default: ``100``.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
        log_level (LogLevel, optional):
            :class:`~.logger.LogLevel` (i.e. unit of resolution) at
            which to record. Default: :attr:`~.LogLevel.EPOCH`.
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 artifact_name: Optional[str] = None,
                 flush_interval: int = 100,
                 rank_zero_only: bool = True,
                 log_level: LogLevel = LogLevel.BATCH):

        self.log_dir = log_dir
        self.artifact_name = artifact_name
        self.flush_interval = flush_interval
        self.rank_zero_only = rank_zero_only
        self.log_level = log_level
        self.writer: SummaryWriter

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        del log_level

        if (not self.rank_zero_only) or dist.get_global_rank() == 0:
            for tag, data_point in data.items():
                if isinstance(data_point, str):  # Will error out with weird caffe2 import error.
                    continue
                try:
                    self.writer.add_scalar(tag, data_point, global_step=int(state.timestamp.batch))
                except NotImplementedError:
                    pass

    def init(self, state: State, logger: Logger) -> None:
        if self.log_dir is None:
            self.log_dir = str(Path.home() / 'tensorboard_logs' / f'{state.run_name}')
        if self.artifact_name is None:
            self.artifact_name = self.log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        #self.should_eval = evaluate_periodically(state.evaluators[0].eval_interval)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.log_level == LogLevel.BATCH and int(state.timestamp.batch) % self.flush_interval == 0:
            self._flush(logger)

    def epoch_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def eval_start(self, state: State, logger: Logger) -> None:
        # Flush any log calls that occurred during INIT when using the trainer in eval-only mode
        self._flush(logger)

    def eval_end(self, state: State, logger: Logger) -> None:
        # Flush any log calls that occurred during INIT when using the trainer in eval-only mode
        self._flush(logger)

    def fit_end(self, state: State, logger: Logger) -> None:
        # Flush the file on fit_end, in case if was not flushed on epoch_end and the trainer is re-used
        # (which would defer when `self.close()` would be invoked)
        self._flush(logger)

    def _flush(self, logger: Logger):
        self.writer.flush()
        logger.file_artifact(LogLevel.FIT,
                             self.artifact_name,
                             file_path=self.writer.file_writer.event_writer._file_name,
                             overwrite=True)
