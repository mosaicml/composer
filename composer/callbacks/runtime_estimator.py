# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate total time of training."""
from __future__ import annotations

from typing import Any, Deque, Dict

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['RuntimeEstimator']

class RuntimeEstimator(Callback):
    """Estimates total training time.

    The training time is computed by taking the time elapsed for the current duration and multiplying
    out to the full extended length of the training run.

    This callback provides a best attempt estimate. This estimate may be inaccurate if throughput
    changes through training or other significant changes are made to the model or dataloader.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import RuntimeEstimator
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[RuntimeEstimator()],
            ... )

    The runtime estimate is logged by the :class:`.Logger` to the following key as described below.

    +----------------------------------+----------------------------------------------------------+
    | Key                              | Logged data                                              |
    +==================================+==========================================================+
    +-----------------------------------+---------------------------------------------------------+
    |                                   | Estimated time to completion using current throughput   |
    | ``wall_clock/remaining_estimate`` | multiplied by remaining time plus a correction for      |
    |                                   | remaining eval calls                                    |
    +-----------------------------------+---------------------------------------------------------+
    """