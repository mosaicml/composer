# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to MPlatform."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

__all__ = ['MPlatformLogger']


class MPlatformLogger(LoggerDestination):
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
        self.time_last_logged = 0
        self.buffered_metrics: Dict[str, Any] = {}

        # TODO: setup msdk

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        self.log_metrics(hyperparameters)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            for key, val in metrics.items():
                self.buffered_metrics[key] = val
            if time.time() - self.time_last_logged > self.log_interval:
                # TODO: log to msdk prefixing with `mosaicml/` and wrapping in try catch
                self.buffered_metrics = {}
                self.time_last_logged = time.time()
