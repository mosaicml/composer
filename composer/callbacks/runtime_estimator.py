# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate total time of training."""
from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

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
    """

    def __init__(self, skip_batches: int = 1) -> None:
        self._enabled = True
        self.batches_left_to_skip = skip_batches
        self.start_time = None
        self.start_dur = None

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.eval_wct_per_label: Dict[str, List[float]] = {}
        # How often eval is called as fraction of total training time
        self.eval_frequency_per_label: Dict[str, float] = {}
        self.last_elapsed_fraction: float = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            'total_eval_wct': self.total_eval_wct,
            'eval_wct_per_label': self.eval_wct_per_label,
            'eval_frequency_per_label': self.eval_frequency_per_label,
            'last_elapsed_fraction': self.last_elapsed_fraction,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.total_eval_wct = state['total_eval_wct']
        self.eval_wct_per_label = state['eval_wct_per_label']
        self.eval_frequency_per_label = state['eval_frequency_per_label']
        self.last_elapsed_fraction = state['last_elapsed_fraction']

    def get_elapsed_duration(self, state: State) -> Optional[float]:
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
        if elapsed_dur > self.start_dur:
            elapsed_time = time.time() - self.start_time
            elapsed_time -= self.total_eval_wct  # Subtract time spent evaluating
            print(
                f'Elapsed time: {elapsed_time}, elapsed duration: {elapsed_dur}, checkpoint duration: {self.start_dur}')
            rate = elapsed_time / (elapsed_dur - self.start_dur)
            remaining_time = rate * (1 - elapsed_dur)

            print(f'Batch end: remaining_time: {remaining_time}')

            # Add remaining time from each evaluator using known frequencies. We explicitly compute
            # frequency instead of using time interpolation to avoid saw tooth pattern in estimates
            for dataloader_label, eval_wcts in self.eval_wct_per_label.items():
                # Discard first eval_wct if possible as it often slower due to dataset downloading
                eval_wct_avg = None
                num_evals_finished = len(eval_wcts)
                if num_evals_finished > 1:
                    eval_wct_avg = sum(eval_wcts[1:]) / (num_evals_finished - 1)
                else:
                    eval_wct_avg = sum(eval_wcts) / num_evals_finished
                eval_rate = self.eval_frequency_per_label[dataloader_label]
                print(
                    f'dataloader_label: {dataloader_label}, eval_wct_avg: {eval_wct_avg}, eval_rate: {eval_rate}, num_evals_finished: {num_evals_finished}'
                )
                if eval_rate > 0:
                    num_total_evals = 1 / eval_rate
                    remaining_calls = num_total_evals - num_evals_finished
                    remaining_time += eval_wct_avg * remaining_calls

            logger.log_metrics({'wall_clock/remaining_estimate': remaining_time})

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
        elapsed_fraction = self.get_elapsed_duration(state)
        if elapsed_fraction is None:
            warnings.warn(
                'Attempting to estimate remaining time but `max_duration` is not set. Skipping adjustment for evaluation time.'
            )
        else:
            self.eval_frequency_per_label[state.dataloader_label] = elapsed_fraction / len(
                self.eval_wct_per_label[state.dataloader_label])
            print(f'Eval finished! eval_frequency_per_label: {self.eval_frequency_per_label[state.dataloader_label]}')
