# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to the MosaicML platform."""

from __future__ import annotations

import collections.abc
import logging
import operator
import os
import time
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

if TYPE_CHECKING:
    from composer.core import State

log = logging.getLogger(__name__)

__all__ = ['MosaicMLLogger']

RUN_NAME_ENV_VAR = 'RUN_NAME'


class MosaicMLLogger(LoggerDestination):
    """Log to the MosaicML platform.

    Logs metrics to the MosaicML platform. Logging only happens on rank 0 every ``log_interval``
    seconds to avoid performance issues.

    Args:
        log_interval (int, optional): Buffer log calls more frequent than ``log_interval`` seconds.
                                      Defaults to 60.
    """

    def __init__(
        self,
        log_interval: int = 60,
    ) -> None:
        self.log_interval = log_interval
        self._enabled = dist.get_global_rank() == 0
        if self._enabled:
            self.time_last_logged = 0
            self.buffered_metadata: Dict[str, Any] = {}

            try:
                import mcli
                del mcli
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='mcli',
                                                    conda_package='mcli',
                                                    conda_channel='conda-forge') from e
            self.run_name = os.environ.get(RUN_NAME_ENV_VAR)
            if self.run_name is not None:
                log.info(f'Logging to mosaic run {self.run_name}')
            else:
                log.warning(f'Environment variable `{RUN_NAME_ENV_VAR}` not set, so MosaicMLLogger '
                            'is disabled as it is unable to identify which run to log to.')
                self._enabled = False

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        self._log_metadata(hyperparameters)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._log_metadata(metrics)

    def batch_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata()

    def epoch_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata()

    def fit_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata()

    def eval_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata()

    def predict_end(self, state: State, logger: Logger) -> None:
        self._flush_metadata()

    def close(self, state: State, logger: Logger) -> None:
        self._flush_metadata()

    def _log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Buffer metadata and prefix keys with mosaicml."""
        if self._enabled:
            for key, val in metadata.items():
                self.buffered_metadata[f'mosaicml/{key}'] = format_data_to_json_serializable(val)
            self._flush_metadata()

    def _flush_metadata(self) -> None:
        """Flush buffered metadata to MosaicML if enough time has passed since last flush."""
        if self._enabled and time.time() - self.time_last_logged > self.log_interval:
            from mcli.api.exceptions import MAPIException
            from mcli.sdk import update_run_metadata
            try:
                update_run_metadata(self.run_name, self.buffered_metadata)
                self.buffered_metadata = {}
                self.time_last_logged = time.time()
            except MAPIException as e:
                log.error(f'Failed to log metadata to Mosaic with error: {e}')


def format_data_to_json_serializable(data: Any):
    """Recursively formats data to be JSON serializable.

    Args:
        data: Data to format.

    Returns:
        str: ``data`` as a string.
    """
    if data is None:
        return 'None'
    if type(data) in (str, int, float, bool):
        return data
    if isinstance(data, torch.Tensor):
        if data.shape == () or reduce(operator.mul, data.shape, 1) == 1:
            return format_data_to_json_serializable(data.cpu().item())
        return 'Tensor of shape ' + str(data.shape)
    if isinstance(data, collections.abc.Mapping):
        return {format_data_to_json_serializable(k): format_data_to_json_serializable(v) for k, v in data.items()}
    if isinstance(data, collections.abc.Iterable):
        return [format_data_to_json_serializable(v) for v in data]

    # Unknown format catch-all
    return str(data)
