# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to Mosaic AI Training."""

from __future__ import annotations

import collections.abc
import fnmatch
import json
import logging
import operator
import os
import time
import warnings
from concurrent.futures import wait
from functools import reduce
from typing import TYPE_CHECKING, Any, Optional

import mcli
import torch

from composer.core.time import TimeUnit
from composer.loggers import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.mlflow_logger import MLFlowLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils import dist

if TYPE_CHECKING:
    from composer.core import State

log = logging.getLogger(__name__)

__all__ = ['MosaicMLLogger', 'MOSAICML_PLATFORM_ENV_VAR', 'MOSAICML_ACCESS_TOKEN_ENV_VAR']

RUN_NAME_ENV_VAR = 'RUN_NAME'
MOSAICML_PLATFORM_ENV_VAR = 'MOSAICML_PLATFORM'
MOSAICML_ACCESS_TOKEN_ENV_VAR = 'MOSAICML_ACCESS_TOKEN_FILE'
MOSAICML_LOG_DIR_ENV_VAR = 'MOSAICML_LOG_DIR'
MOSAICML_GPU_LOG_FILE_PREFIX_ENV_VAR = 'MOSAICML_GPU_LOG_FILE_PREFIX'


class MosaicMLLogger(LoggerDestination):
    """Log to Mosaic AI Training.

    Logs metrics to Mosaic AI Training. Logging only happens on rank 0 every ``log_interval``
    seconds to avoid performance issues.

    When running on Mosaic AI Training, the logger is automatically enabled by Trainer. To disable,
    the environment variable 'MOSAICML_PLATFORM' can be set to False.

    Args:
        log_interval (int, optional): Buffer log calls more frequent than ``log_interval`` seconds
            to avoid performance issues. Defaults to 60.
        ignore_keys (list[str], optional): A list of keys to ignore when logging. The keys support
            Unix shell-style wildcards with fnmatch. Defaults to ``None``.

            Example 1: ``ignore_keys = ["wall_clock/train", "wall_clock/val", "wall_clock/total"]``
            would ignore wall clock metrics.

            Example 2: ``ignore_keys = ["wall_clock/*"]`` would ignore all wall clock metrics.

            (default: ``None``)
        ignore_exceptions: Flag to disable logging exceptions. Defaults to True.
    """

    def __init__(
        self,
        log_interval: int = 60,
        ignore_keys: Optional[list[str]] = None,
        ignore_exceptions: bool = True,
    ) -> None:
        self.log_interval = log_interval
        self.ignore_keys = ignore_keys
        self.ignore_exceptions = ignore_exceptions
        self._enabled = dist.get_global_rank() == 0
        if self._enabled:
            self.time_last_logged = 0
            self.train_dataloader_len = None
            self.buffered_metadata: dict[str, Any] = {}
            self._futures = []

            self.run_name = os.environ.get(RUN_NAME_ENV_VAR)
            if self.run_name is not None:
                log.info(f'Logging to mosaic run {self.run_name}')
            else:
                log.warning(
                    f'Environment variable `{RUN_NAME_ENV_VAR}` not set, so MosaicMLLogger '
                    'is disabled as it is unable to identify which run to log to.',
                )
                self._enabled = False

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        self.log_metadata(hyperparameters)

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self.log_metadata(metrics)

    def after_load(self, state: State, logger: Logger) -> None:
        # If model was resumed from checkpoint, log the checkpoint path
        if int(state.timestamp.batch) > 0 and state.load_path is not None:
            log.debug(f'Logging checkpoint path to metadata: {state.load_path}')
            self.log_metadata({'checkpoint_resumed_path': state.load_path})

        # Log model data downloaded and initialized for run events
        log.debug(f'Logging model initialized time to metadata')
        self.log_metadata({'model_initialized_time': time.time()})
        # Log WandB run URL if it exists. Must run on after_load as WandB is setup on event init
        for callback in state.callbacks:
            if isinstance(callback, WandBLogger):
                run_url = callback.run_url
                if run_url is not None:
                    self.log_metadata({'wandb/run_url': run_url})
                    log.debug(f'Logging WandB run URL to metadata: {run_url}')
                else:
                    log.debug('WandB run URL not found, not logging to metadata')
            if isinstance(callback, MLFlowLogger) and callback._enabled:
                self.log_metadata({'mlflow/run_url': callback.run_url})
                log.debug(f'Logging MLFlow run URL to metadata: {callback.run_url}')
        self._flush_metadata(force_flush=True)

    def batch_start(self, state: State, logger: Logger) -> None:
        if state.dataloader_len is not None and self._enabled:
            self.train_dataloader_len = state.dataloader_len.value

    def batch_end(self, state: State, logger: Logger) -> None:
        training_progress_data = self._get_training_progress_metrics(state)
        self.log_metadata(training_progress_data)
        self._flush_metadata()

    def epoch_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata()

    def fit_end(self, state: State, logger: Logger) -> None:
        # Log model training finished time for run events
        self.log_metadata({'train_finished_time': time.time()})
        training_progress_data = self._get_training_progress_metrics(state)
        log.debug(f'\nLogging FINAL training progress data to metadata:\n{dict_to_str(training_progress_data)}')
        self.log_metadata(training_progress_data)
        self._flush_metadata(force_flush=True)

    def fit_start(self, state: State, logger: Logger) -> None:
        # Log model training started time for run events
        self.log_metadata({'train_started_time': time.time()})
        self._flush_metadata(force_flush=True)

    def eval_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata(force_flush=True)

    def predict_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata(force_flush=True)

    def close(self, state: State, logger: Logger) -> None:
        # Skip flushing metadata as it should be logged by fit/eval/predict_end. Flushing here
        # might schedule futures while interpreter is shutting down, which will raise an error.
        if self._enabled:
            wait(self._futures)  # Ignore raised errors on close

    def log_metadata(self, metadata: dict[str, Any], force_flush: bool = False) -> None:
        """Buffer metadata and prefix keys with mosaicml."""
        if self._enabled:
            for key, val in metadata.items():
                if self.ignore_keys and any(fnmatch.fnmatch(key, pattern) for pattern in self.ignore_keys):
                    continue
                self.buffered_metadata[f'mosaicml/{key}'] = format_data_to_json_serializable(val)
            self._flush_metadata(force_flush=force_flush)

    def _flush_metadata(self, force_flush: bool = False, future: bool = True) -> None:
        """Flush buffered metadata to MosaicML if enough time has passed since last flush."""
        if self._enabled and len(
            self.buffered_metadata,
        ) > 0 and (time.time() - self.time_last_logged > self.log_interval or force_flush):
            try:
                assert self.run_name is not None
                if future:
                    f = mcli.update_run_metadata(self.run_name, self.buffered_metadata, future=True, protect=True)
                    self._futures.append(f)
                else:
                    mcli.update_run_metadata(self.run_name, self.buffered_metadata, future=False, protect=True)
                self.buffered_metadata = {}
                self.time_last_logged = time.time()
                done, incomplete = wait(self._futures, timeout=0.01)
                # Raise any exceptions
                for f in done:
                    if f.exception() is not None:
                        raise f.exception()  # type: ignore
                self._futures = list(incomplete)
            except Exception:
                log.exception('Failed to log metadata to Mosaic')  # Prints out full traceback
                if self.ignore_exceptions:
                    log.info('Ignoring exception and disabling MosaicMLLogger.')
                    self._enabled = False
                else:
                    log.info('Raising exception. To ignore exceptions, set ignore_exceptions=True.')
                    raise

    def _get_training_progress_metrics(self, state: State) -> dict[str, Any]:
        """Calculates training progress metrics.

        If user submits max duration:
        - in tokens -> format: [token=x/xx]
        - in batches -> format: [batch=x/xx]
        - in epoch -> format: [epoch=x/xx] [batch=x/xx] (where batch refers to batches completed in current epoch)
        If batches per epoch cannot be calculated, return [epoch=x/xx]

        If no training duration given -> format: ''
        """
        if not self._enabled:
            return {}

        assert state.max_duration is not None
        if state.max_duration.unit == TimeUnit.TOKEN:
            return {
                'training_progress': f'[token={state.timestamp.token.value}/{state.max_duration.value}]',
            }
        if state.max_duration.unit == TimeUnit.BATCH:
            return {
                'training_progress': f'[batch={state.timestamp.batch.value}/{state.max_duration.value}]',
            }
        training_progress_metrics = {}
        if state.max_duration.unit == TimeUnit.EPOCH:
            cur_batch = state.timestamp.batch_in_epoch.value
            cur_epoch = state.timestamp.epoch.value
            if state.timestamp.epoch.value >= 1:
                batches_per_epoch = (
                    state.timestamp.batch - state.timestamp.batch_in_epoch
                ).value // state.timestamp.epoch.value
                curr_progress = f'[batch={cur_batch}/{batches_per_epoch}]'
            elif self.train_dataloader_len is not None:
                curr_progress = f'[batch={cur_batch}/{self.train_dataloader_len}]'
            else:
                curr_progress = f'[batch={cur_batch}]'
            if cur_epoch < state.max_duration.value:
                cur_epoch += 1
            training_progress_metrics = {
                'training_sub_progress': curr_progress,
                'training_progress': f'[epoch={cur_epoch}/{state.max_duration.value}]',
            }
        return training_progress_metrics


def format_data_to_json_serializable(data: Any):
    """Recursively formats data to be JSON serializable.

    Args:
        data: Data to format.

    Returns:
        str: ``data`` as a string.
    """
    try:
        ret = None
        if data is None:
            ret = 'None'
        elif type(data) in (str, int, float, bool):
            ret = data
        elif isinstance(data, torch.Tensor):
            if data.shape == () or reduce(operator.mul, data.shape, 1) == 1:
                ret = format_data_to_json_serializable(data.cpu().item())
            else:
                ret = 'Tensor of shape ' + str(data.shape)
        elif isinstance(data, collections.abc.Mapping):
            ret = {format_data_to_json_serializable(k): format_data_to_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, collections.abc.Iterable):
            ret = [format_data_to_json_serializable(v) for v in data]
        else:  # Unknown format catch-all
            ret = str(data)
        json.dumps(ret)  # Check if ret is JSON serializable
        return ret
    except RuntimeError as e:
        warnings.warn(
            f'Encountered unexpected error while formatting data of type {type(data)} to '
            f'be JSON serializable. Returning empty string instead. Error: {str(e)}',
        )
        return ''


def dict_to_str(data: dict[str, Any]):
    return '\n'.join([f'\t{k}: {v}' for k, v in data.items()])


def exception_to_json_serializable_dict(exc: Exception):
    """Converts exception into a JSON serializable dictionary for run metadata."""
    default_exc_attrs = set(dir(Exception()))
    exc_data = {'class': exc.__class__.__name__, 'message': str(exc), 'attributes': {}}

    for attr in dir(exc):
        # Exclude default attributes and special methods
        if attr not in default_exc_attrs and not attr.startswith('__'):
            try:
                value = getattr(exc, attr)
                if callable(value):
                    continue
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    exc_data['attributes'][attr] = value
                else:
                    exc_data['attributes'][attr] = str(value)
            except AttributeError:
                pass
    return exc_data
