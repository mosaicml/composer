# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate total time of training."""
from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional

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
    | `wall_clock/remaining_estimate`   | Estimated time to completion                            |
    +-----------------------------------+---------------------------------------------------------+

    Args:
        skip_batches (int, optional): Number of batches to skip before starting clock to estimate
            remaining time. Typically, the first few batches are slower due to dataloader, cache
            warming, and other reasons. Defaults to 1.
        time_unit (str, optional): Time unit to use for `wall_clock` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.
    """

    def __init__(self, skip_batches: int = 1, time_unit: str = 'hours') -> None:
        self._enabled = True
        self.batches_left_to_skip = skip_batches
        self.start_time = None
        self.start_dur = None

        self.divider = 1
        if time_unit == 'seconds':
            self.divider = 1
        elif time_unit == 'minutes':
            self.divider = 60
        elif time_unit == 'hours':
            self.divider = 60 * 60
        elif time_unit == 'days':
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".')

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.eval_wct_per_label: Dict[str, List[float]] = {}
        # How often eval is called as fraction of total training time
        self.eval_frequency_per_label: Dict[str, float] = {}
        self.last_elapsed_fraction: float = 0.0

    def _get_elapsed_duration(self, state: State) -> Optional[float]:
        """Get the elapsed duration.

        Unlike `state.get_elapsed_duration`, this method computes fractional progress in an epoch
        provided at least 1 epoch has passed by recording how many batches were in each epoch.
        """
        if state.max_duration is None:
            return None
        if state.max_duration.unit == TimeUnit('ep'):
            if state.timestamp.epoch.value >= 1:
                batches_per_epoch = (state.timestamp.batch -
                                     state.timestamp.batch_in_epoch).value / state.timestamp.epoch.value
                return state.timestamp.get('ba').value / (state.max_duration.value * batches_per_epoch)
            elif state.dataloader_len is not None:
                return state.timestamp.get('ba').value / (state.max_duration.value * state.dataloader_len.value)
        elapsed_dur = state.get_elapsed_duration()
        if elapsed_dur is not None:
            return elapsed_dur.value
        return None

    def batch_start(self, state: State, logger: Logger) -> None:
        if self._enabled and self.start_time is None and self.batches_left_to_skip == 0:
            self.start_time = time.time()
            self.start_dur = self._get_elapsed_duration(state)
            if self.start_dur is None:
                warnings.warn('`max_duration` is not set. Cannot estimate remaining time.')
                self._enabled = False

    def batch_end(self, state: State, logger: Logger) -> None:
        if not self._enabled:
            return
        if self.batches_left_to_skip > 0:
            self.batches_left_to_skip -= 1
            return

        elapsed_dur = self._get_elapsed_duration(state)
        assert elapsed_dur is not None, 'max_duration checked as non-None on batch_start if enabled'

        assert self.start_dur is not None
        assert self.start_time is not None
        if elapsed_dur > self.start_dur:
            elapsed_time = time.time() - self.start_time
            elapsed_time -= self.total_eval_wct  # Subtract time spent evaluating
            rate = elapsed_time / (elapsed_dur - self.start_dur)
            remaining_time = rate * (1 - elapsed_dur)

            # Add remaining time from each evaluator using known frequencies. We explicitly compute
            # frequency instead of using time interpolation to avoid saw tooth pattern in estimates
            for dataloader_label, eval_wcts in self.eval_wct_per_label.items():
                # Discard first eval_wct if possible as it is often slower due to dataset downloading
                eval_wct_avg = None
                num_evals_finished = len(eval_wcts)
                if num_evals_finished > 1:
                    eval_wct_avg = sum(eval_wcts[1:]) / (num_evals_finished - 1)
                else:
                    eval_wct_avg = sum(eval_wcts) / num_evals_finished
                eval_rate = self.eval_frequency_per_label[dataloader_label]
                num_total_evals = 1 / eval_rate
                remaining_calls = num_total_evals - num_evals_finished
                remaining_time += eval_wct_avg * remaining_calls

            logger.log_metrics({'wall_clock/remaining_estimate': remaining_time / self.divider})

    def eval_end(self, state: State, logger: Logger) -> None:
        # If eval is called before training starts, ignore it
        if not self._enabled or self.start_time is None:
            return
        self.total_eval_wct += state.eval_timestamp.total_wct.total_seconds()
        # state.dataloader_label should always be non-None unless user explicitly sets evaluator
        # label to None, ignoring type hints
        assert state.dataloader_label is not None, 'evaluator label must not be None'
        if state.dataloader_label not in self.eval_wct_per_label:
            self.eval_wct_per_label[state.dataloader_label] = []
        self.eval_wct_per_label[state.dataloader_label].append(state.eval_timestamp.total_wct.total_seconds())
        elapsed_dur = self._get_elapsed_duration(state)
        assert elapsed_dur is not None, 'max_duration checked as non-None on batch_start if enabled'
        assert self.start_dur is not None, 'start_dur is set on batch_start if enabled'
        elapsed_fraction = elapsed_dur - self.start_dur
        num_evals_finished = len(self.eval_wct_per_label[state.dataloader_label])
        self.eval_frequency_per_label[state.dataloader_label] = elapsed_fraction / num_evals_finished
