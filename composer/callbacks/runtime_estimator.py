# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate total time of training."""
from __future__ import annotations

import time
import warnings
from typing import Optional

from composer.core import Callback, State, TimeUnit
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

    +-----------------------------------+---------------------------------------------------------+
    | Key                               | Logged data                                             |
    +===================================+=========================================================+
    | ``wall_clock/remaining_estimate`` | Estimated time to completion                            |
    +-----------------------------------+---------------------------------------------------------+

    Args:
        skip_batches (int, optional): Number of batches to skip before starting clock to estimate
            remaining time. Typically, the first few batches are slower due to dataloader, cache
            warming, and other reasons. Defaults to 1.
    """

    def __init__(self, skip_batches: int = 1) -> None:
        self._enabled = True
        self.batches_left_to_skip = skip_batches
        self.start_time = None
        self.start_dur = None

    def get_elapsed_duration(self, state: State) -> Optional[float]:
        """Get the elapsed duration.

        Unlike `state.get_elapsed_duration`, this method computes fractional progress in an epoch
        provided at least 1 epoch has passed by recording how many batches were in each epoch.
        """
        if state.max_duration is None:
            return None
        if state.max_duration.unit == TimeUnit('ep') and state.timestamp.epoch.value >= 1:
            batches_per_epoch = (state.timestamp.batch -
                                 state.timestamp.batch_in_epoch).value / state.timestamp.epoch.value
            return state.timestamp.get('ba').value / (state.max_duration.value * batches_per_epoch)
        # TODO: Compute ba count if max dur in ep but dataloader len is available
        elapsed_dur = state.get_elapsed_duration()
        if elapsed_dur is not None:
            return elapsed_dur.value
        return None

    def batch_start(self, state: State, logger: Logger) -> None:
        if self._enabled and self.start_time is None and self.batches_left_to_skip == 0:
            self.start_time = time.time()
            self.start_dur = self.get_elapsed_duration(state)
            if self.start_dur is None:
                warnings.warn('`max_duration` is not set. Cannot estimate remaining time.')
                self._enabled = False

    def batch_end(self, state: State, logger: Logger) -> None:
        if not self._enabled:
            return
        if self.batches_left_to_skip > 0:
            self.batches_left_to_skip -= 1
            return

        elapsed_dur = self.get_elapsed_duration(state)
        if elapsed_dur is None:
            self._enabled = False
            warnings.warn('`max_duration` is not set. Cannot estimate remaining time.')
            return

        assert self.start_dur is not None
        assert self.start_time is not None
        elapsed_time = time.time() - self.start_time
        print(f'Elapsed time: {elapsed_time}, elapsed duration: {elapsed_dur}, checkpoint duration: {self.start_dur}')
        rate = elapsed_time / (elapsed_dur - self.start_dur)
        remaining_time = rate * (1 - elapsed_dur)
        logger.log_metrics({'wall_clock/remaining_estimate': remaining_time})
