# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Tensorboard <https://www.tensorflow.org/tensorboard/>`_."""

from pathlib import Path
from typing import Any, Dict, Optional

from composer.core.state import State
from composer.loggers.logger import Logger, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

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
        self.run_name: Optional[str] = None
        self.hyperparameters: Dict[str, Any] = {}
        self.current_metrics: Dict[str, Any] = {}

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):

        if self.rank_zero_only and dist.get_global_rank() != 0:
            return
        # Lazy logging of hyperparameters b/c Tensorboard requires a metric to pair
        # with hyperparameters.
        formatted_hparams = {
            hparam_name: format_log_data_value(hparam_value) for hparam_name, hparam_value in hyperparameters.items()
        }
        self.hyperparameters.update(formatted_hparams)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.rank_zero_only and dist.get_global_rank() != 0:
            return

        # Keep track of most recent metrics to use for `add_hparams` call.
        self.current_metrics.update(metrics)

        for tag, metric in metrics.items():
            if isinstance(metric, str):  # Will error out with weird caffe2 import error.
                continue
            # TODO: handle logging non-(scalars/arrays/tensors/strings)
            # If a non-(scalars/arrays/tensors/strings) is passed, we skip logging it,
            # so that we do not crash the job.
            try:
                assert self.writer is not None
                self.writer.add_scalar(tag, metric, global_step=step)
            # Gets raised if data_point is not a tensor, array, scalar, or string.
            except NotImplementedError:
                pass

    def init(self, state: State, logger: Logger) -> None:
        self.run_name = state.run_name

        # We fix the log_dir, so all runs are co-located.
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

        # Disable SummaryWriter's internal flushing to avoid file corruption while
        # file staged for upload to an ObjectStore.
        flush_secs = 365 * 3600 * 24
        self.writer = SummaryWriter(log_dir=summary_writer_log_dir, flush_secs=flush_secs)

    def batch_end(self, state: State, logger: Logger) -> None:
        if int(state.timestamp.batch) % self.flush_interval == 0:
            self._flush(logger)

    def epoch_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def eval_end(self, state: State, logger: Logger) -> None:
        # Give the metrics used for hparams a unique name, so they don't get plotted in the
        # normal metrics plot.
        metrics_for_hparams = {
            'hparams/' + name: metric
            for name, metric in self.current_metrics.items()
            if 'metric' in name or 'loss' in name
        }
        assert self.writer is not None
        self.writer.add_hparams(hparam_dict=self.hyperparameters,
                                metric_dict=metrics_for_hparams,
                                run_name=self.run_name)
        self._flush(logger)

    def fit_end(self, state: State, logger: Logger) -> None:
        self._flush(logger)

    def _flush(self, logger: Logger):
        # To avoid empty files uploaded for each rank.
        if self.rank_zero_only and dist.get_global_rank() != 0:
            return

        if self.writer is None:
            return
        # Skip if no writes occurred since last flush.
        if not self.writer.file_writer:
            return

        self.writer.flush()

        file_path = self.writer.file_writer.event_writer._file_name
        event_file_name = Path(file_path).stem

        logger.upload_file(remote_file_name=('tensorboard_logs/{run_name}/' +
                                             f'{event_file_name}-{dist.get_global_rank()}'),
                           file_path=file_path,
                           overwrite=True)

        # Close writer, which creates new log file.
        self.writer.close()

    def close(self, state: State, logger: Logger) -> None:
        del state  # unused
        self._flush(logger)
        self.writer = None
