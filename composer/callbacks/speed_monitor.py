# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from composer.core import Callback, State, TimeUnit
from composer.loggers import Logger

__all__ = ['SpeedMonitor']


class SpeedMonitor(Callback):
    """Logs the training throughput.

    The training throughput in terms of number of samples per second is logged on the
    :attr:`.Event.BATCH_END` event if we have reached the ``window_size`` threshold.

    The wall clock train time is logged on every :attr:`.Event.BATCH_END` event.

    The average throughout over an epoch is logged on the :attr:`.Event.EPOCH_END` event.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import SpeedMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[SpeedMonitor(window_size=100)],
            ... )

    The training throughput is logged by the :class:`.Logger` to the following keys as
    described below.

    +-----------------------------------+---------------------------------------------------------+
    | Key                               | Logged data                                             |
    +===================================+=========================================================+
    |                                   | Rolling average (over ``window_size`` most recent       |
    | ``throughput/samples_per_sec``    | batches) of the number of samples processed per second  |
    |                                   |                                                         |
    +-----------------------------------+---------------------------------------------------------+
    | ``wall_clock/train``              | Total elapsed training time                             |
    +-----------------------------------+---------------------------------------------------------+
    | ``wall_clock/val``                | Total elapsed validation time                           |
    +-----------------------------------+---------------------------------------------------------+
    | ``wall_clock/total``              | Total elapsed time (wall_clock/train + wall_clock/val)  |
    +-----------------------------------+---------------------------------------------------------+
    |                                   | Estimated time to completion using estimated throughput |
    | ``wall_clock/remaining_estimate`` | multiplied by remaining time plus a correction for      |
    |                                   | remaining eval calls                                    |
    +-----------------------------------+---------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
    """

    def __init__(self, window_size: int = 100):
        # Track the batch num samples and wct to compute throughput over a window of batches
        self.batch_start_num_samples = 0
        self.batch_start_wct = 0.0
        self.batch_wct_buffer: Deque[float] = deque(maxlen=window_size)
        self.batch_num_samples_buffer: Deque[int] = deque(maxlen=window_size)
        self.window_size = window_size

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.eval_wct_per_label: Dict[str, List[float]] = {}
        # How often eval is called as fraction of total training time
        self.eval_frequency_per_label: Dict[str, float] = {}
        self.last_elapsed_fraction: float = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            'batch_start_num_samples': self.batch_start_num_samples,
            'batch_start_wct': self.batch_start_wct,
            'batch_wct_buffer': self.batch_wct_buffer,
            'batch_num_samples_buffer': self.batch_num_samples_buffer,
            'total_eval_wct': self.total_eval_wct,
            'eval_wct_per_label': self.eval_wct_per_label,
            'eval_frequency_per_label': self.eval_frequency_per_label,
            'last_elapsed_fraction': self.last_elapsed_fraction,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.batch_start_num_samples = state['batch_start_num_samples']
        self.batch_start_wct = state['batch_start_wct']
        self.batch_wct_buffer = deque(
            [x for x in state['batch_wct_buffer']],
            maxlen=self.window_size,
        )
        self.batch_num_samples_buffer = deque(
            [x for x in state['batch_num_samples_buffer']],
            maxlen=self.window_size,
        )
        self.total_eval_wct = state['total_eval_wct']

        # Added in 0.13. Load only if present to support backwards compatibility
        self.eval_wct_per_label = state.get('eval_wct_per_label', self.eval_wct_per_label)
        self.eval_frequency_per_label = state.get('eval_frequency_per_label', self.eval_frequency_per_label)
        self.last_elapsed_fraction = state.get('last_elapsed_fraction', self.last_elapsed_fraction)

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
        elapsed_dur = state.get_elapsed_duration()
        if elapsed_dur is not None:
            return elapsed_dur.value
        return None

    def before_dataloader(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_wct = state.timestamp.total_wct.total_seconds()
        self.batch_start_num_samples = int(state.timestamp.sample)

    def batch_end(self, state: State, logger: Logger):
        batch_num_samples = int(state.timestamp.sample) - self.batch_start_num_samples
        batch_wct = state.timestamp.total_wct.total_seconds() - self.batch_start_wct

        # Add the new element
        self.batch_wct_buffer.append(batch_wct)
        self.batch_num_samples_buffer.append(batch_num_samples)

        if len(self.batch_num_samples_buffer) == self.window_size:
            # Log the throughput
            throughput = sum(self.batch_num_samples_buffer) / sum(self.batch_wct_buffer)
            logger.log_metrics({'throughput/samples_per_sec': throughput})

            # Estimate remaining time
            elapsed_fraction = self.get_elapsed_duration(state)
            print(throughput, elapsed_fraction)
            if elapsed_fraction is None:
                warnings.warn('`max_duration` is not set. Cannot estimate remaining time.')
            elif elapsed_fraction > 0 and elapsed_fraction != self.last_elapsed_fraction:
                # Only update the estimate if the elapsed duration has changed
                self.last_elapsed_fraction = elapsed_fraction
                batch_wct_avg = sum(self.batch_wct_buffer) / len(self.batch_wct_buffer)
                total_num_batches = int(state.timestamp.batch) / elapsed_fraction
                remaining_time = batch_wct_avg * total_num_batches * (1 - elapsed_fraction)
                # Add remaining time from each evaluator
                for dataloader_label, eval_wcts in self.eval_wct_per_label.items():
                    # Discard first eval_wct if possible as it often slower due to dataset downloading
                    eval_wct_avg = None
                    num_evals_finished = len(eval_wcts)
                    if num_evals_finished > 1:
                        eval_wct_avg = sum(eval_wcts[1:]) / (num_evals_finished - 1)
                    else:
                        eval_wct_avg = sum(eval_wcts) / num_evals_finished
                    eval_rate = self.eval_frequency_per_label[dataloader_label]
                    if eval_rate > 0:
                        num_total_evals = 1 / eval_rate
                        remaining_calls = num_total_evals - num_evals_finished
                        remaining_time += eval_wct_avg * remaining_calls
                logger.log_metrics({'wall_clock/remaining_estimate': remaining_time})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'wall_clock/train': state.timestamp.total_wct.total_seconds(),
            'wall_clock/val': self.total_eval_wct,
            'wall_clock/total': (state.timestamp.total_wct.total_seconds() + self.total_eval_wct),
        })

    def eval_end(self, state: State, logger: Logger):
        del logger  # unused
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
