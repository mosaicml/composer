# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Tensorboard <https://www.tensorflow.org/tensorboard/>`_."""

from pathlib import Path
from typing import Any, Dict, Optional

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['TensorboardLogger']


class TensorboardLogger(LoggerDestination):
    """Log to `Tensorboard <https://www.tensorflow.org/tensorboard/>`_.

    If you are accessing your logs from a cloud bucket, like S3, they will be
    in `{your_bucket_name}/tensorboard_logs/{run_name}` with names like
    `events.out.tfevents-{run_name}-{rank}`.

    If you are accessing your logs locally (from wherever you are running composer), the logs
    will be in the relative path: `tensorboard_logs/{run_name}` with names starting with
    `events.out.tfevents.*`

    Args:
        log_dir (str, optional): The path to the directory where all the tensorboard logs
            will be saved. This is also the value that should be specified when starting
            a tensorboard server. e.g. `tensorboard --logdir={log_dir}`. If not specified
            `./tensorboard_logs` will be used.
        flush_interval (int, optional): How frequently by batch to flush the log to a file.
            For example, a flush interval of 10 means the log will be flushed to a file
            every 10 batches. The logs will also be automatically flushed at the start and
            end of every evaluation phase (`EVENT.EVAL_START` and `EVENT.EVAL_END` ),
            the end of every epoch (`EVENT.EPOCH_END`), and the end of training
            (`EVENT.FIT_END`). Default: ``100``.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            Recommended to be true since the rank 0 will have access to most global metrics.
            A setting of `False` may lead to logging of duplicate values.
            Default: :attr:`True`.
    """

    def __init__(self, log_dir: Optional[str] = None, flush_interval: int = 100, rank_zero_only: bool = True):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='tensorboard',
                                                conda_package='tensorboard',
                                                conda_channel='conda-forge') from e

        self.log_dir = log_dir
        self.flush_interval = flush_interval
        self.rank_zero_only = rank_zero_only
        self.writer: Optional[SummaryWriter] = None

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        del log_level

        if self.rank_zero_only and dist.get_global_rank() != 0:
            return
        for tag, data_point in data.items():
            if isinstance(data_point, str):  # Will error out with weird caffe2 import error.
                continue
            # TODO: handle logging non-(scalars/arrays/tensors/strings)
            # If a non-(scalars/arrays/tensors/strings) is passed, we skip logging it,
            # so that we do not crash the job.
            try:
                assert self.writer is not None
                self.writer.add_scalar(tag, data_point, global_step=int(state.timestamp.batch))
            # Gets raised if data_point is not a tensor, array, scalar, or string.
            except NotImplementedError:
                pass

    def init(self, state: State, logger: Logger) -> None:
        from torch.utils.tensorboard import SummaryWriter

        # We set the log_dir to a constant, so all runs can be co-located together.
        if self.log_dir is None:
            self.log_dir = 'tensorboard_logs'

        # We name the child directory after the run_name to ensure the run_name shows up
        # in the Tensorboard GUI.
        summary_writer_log_dir = Path(self.log_dir) / state.run_name

        # To disable automatic flushing we set flushing to once a year ;)
        flush_secs = 365 * 3600 * 24
        self.writer = SummaryWriter(log_dir=summary_writer_log_dir, flush_secs=flush_secs)

    def batch_end(self, state: State, logger: Logger) -> None:
        if int(state.timestamp.batch) % self.flush_interval == 0:
            self._flush(logger)

    def epoch_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def eval_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def fit_end(self, state: State, logger: Logger) -> None:
        # Flush the file on fit_end, in case if was not flushed on epoch_end and the trainer is re-used
        # (which would defer when `self.close()` would be invoked)
        self._flush(logger)

    def _flush(self, logger: Logger):
        # To avoid empty log artifacts for each rank.
        if self.rank_zero_only and dist.get_global_rank() != 0:
            return

        assert self.writer is not None
        self.writer.flush()

        assert self.writer.file_writer is not None
        file_path = self.writer.file_writer.event_writer._file_name

        logger.file_artifact(
            LogLevel.FIT,
            # For a file to be readable by Tensorboard, it must start with
            # 'events.out.tfevents'. Child directory is named after run_name, so the logs
            # are named properly in the Tensorboard GUI.
            artifact_name='tensorboard_logs/{run_name}/events.out.tfevents-{run_name}-{rank}',
            file_path=file_path,
            overwrite=True)
