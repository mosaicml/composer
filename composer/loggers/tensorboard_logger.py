# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Tensorboard <https://www.tensorflow.org/tensorboard/>`_."""

import time
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
        flush_secs (int, optional): How frequently in seconds to flush the log to a file
            at the end of batch. Specifically, a log is flushed at the end of batch only
            if it has been `flush_secs` since the last flush. The logs will also be 
            automatically flushed at the end of every evaluation phase (`EVENT.EVAL_END`),
             the end of every epoch (`EVENT.EPOCH_END`), and the end of training
             (`EVENT.FIT_END`). Default: ``20``.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            Recommended to be true since the rank 0 will have access to most global metrics.
            A setting of `False` may lead to logging of duplicate values.
            Default: :attr:`True`.
    """

    def __init__(self, log_dir: Optional[str] = None, flush_secs: int = 20, rank_zero_only: bool = True):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='tensorboard',
                                                conda_package='tensorboard',
                                                conda_channel='conda-forge') from e

        self.log_dir = log_dir
        self.rank_zero_only = rank_zero_only
        self.writer: Optional[SummaryWriter] = None
        self.flush_count = 0
        self.event_file_base_file_path: Optional[str] = None
        self.flush_secs = flush_secs
        self.last_flush = time.time()
        self.run_name: Optional[str] = None

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
        self.run_name = state.run_name

        # We set the log_dir to a constant, so all runs can be co-located together.
        if self.log_dir is None:
            self.log_dir = 'tensorboard_logs'

        self._initialize_summary_writer()

    def _initialize_summary_writer(self):
        from torch.utils.tensorboard import SummaryWriter

        assert self.run_name is not None
        assert self.log_dir is not None
        # We name the child directory after the run_name to ensure the run_name shows up
        # in the Tensorboard GUI.
        summary_writer_log_dir = Path(self.log_dir) / self.run_name

        # To disable automatic flushing we set flushing to once a year ;)
        summary_writer_flush_secs = 365 * 3600 * 24
        self.writer = SummaryWriter(log_dir=summary_writer_log_dir, flush_secs=summary_writer_flush_secs)

        if self.event_file_base_file_path is None:
            self.event_file_base_file_path = self.writer.file_writer.event_writer._file_name

        # Give event_file a unique name to avoid appending to the same file on every flush.
        self.writer.file_writer.event_writer._file_name = self.event_file_base_file_path + f'-{self.flush_count}'
        self.writer.file_writer.event_writer._async_writer._writer._writer.filename = self.writer.file_writer.event_writer._file_name


    def batch_end(self, state: State, logger: Logger) -> None:
        if (time.time() - self.last_flush) >= self.flush_secs:
            self._flush(logger)

    def epoch_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def eval_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def fit_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def _flush(self, logger: Logger):
        # To avoid empty log artifacts for each rank.
        if self.rank_zero_only and dist.get_global_rank() != 0:
            return

        assert self.writer.file_writer is not None
        file_path = self.writer.file_writer.event_writer._file_name

        # If no writes have happened since the last flush, then file_path won't exist, so
        # we should skip doing flushing and skip filing the artifact since it will error
        # out anyway (given that the file_path doesn't exist).
        if not Path(file_path).exists():
            return

        assert self.writer is not None
        self.writer.flush()
        self.last_flush = time.time()

     
        logger.file_artifact(
            LogLevel.FIT,
            # For a file to be readable by Tensorboard, it must start with
            # 'events.out.tfevents'. Child directory is named after run_name, so the logs
            # are named properly in the Tensorboard GUI.
            artifact_name=(
                'tensorboard_logs/{run_name}/events.out.tfevents-{run_name}-{rank}'
                # Append flush_count, so artifact has unique name.
                + f'-{self.flush_count}'),
            file_path=file_path,
            overwrite=True)

        # Close writer and reinitialize it with a new log file path. This ensures that 
        # we have one file per flush, which means `file_artifact` is always copying 
        # data points that have never been copied before. Also this ensures no strange 
        # issues with dropping of points when setting a new log file path.
        self.writer.close()
        self.flush_count += 1
        self._initialize_summary_writer()
