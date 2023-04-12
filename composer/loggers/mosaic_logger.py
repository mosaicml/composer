# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to MPlatform."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from composer.loggers.logger import format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

log = logging.getLogger(__name__)

__all__ = ['MosaicLogger']


class MosaicLogger(LoggerDestination):
    """Log to MPlatform.

    Logs metrics to MPlatform. Logging only happens on rank 0 every ``log_interval`` seconds to
    avoid performance issues.

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
            self.run_name = os.environ.get('RUN_NAME')
            if self.run_name is not None:
                log.info(f'Logging to mosaic run {self.run_name}')
            else:
                log.warning('Environment variable `RUN_NAME` not set, so MosaicLogger is disabled '
                            'as it is unable to identify which run to log to.')
                self._enabled = False

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        self._log_metadata(hyperparameters)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._log_metadata(metrics)

    def _log_metadata(self, metadata: Dict[str, Any]) -> None:
        if self._enabled:
            for key, val in metadata.items():
                self.buffered_metadata[key] = format_log_data_value(val)
            if time.time() - self.time_last_logged > self.log_interval:
                from mcli.api.exceptions import MAPIException
                from mcli.sdk import update_run_metadata
                try:
                    print(f'\n\nLogging metadata to Mosaic: {self.buffered_metadata}')
                    update_run_metadata(self.run_name, self.buffered_metadata)
                except MAPIException as e:
                    log.error(f'Failed to log metadata to Mosaic with error: {e}')

                self.buffered_metadata = {}
                self.time_last_logged = time.time()
