# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

import datetime
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from composer.core import State, Time
from composer.core.callback import Callback
from composer.loggers import Logger

__all__ = ["SpeedMonitor"]


class SpeedMonitor(Callback):
    """Logs the training throughput.

    The training throughput in terms of number of samples per second is logged on
    the :attr:`~composer.core.event.Event.BATCH_END` event if we have reached the ``window_size`` threshold. Per epoch
    average throughput and wall clock train, validation, and total time is also logged on
    the :attr:`~composer.core.event.Event.EPOCH_END` event.

    Example:
    .. doctest::

        >>> from composer.callbacks import SpeedMonitor
        >>> # constructing trainer object with this callback
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     eval_dataloader=eval_dataloader,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[SpeedMonitor(window_size=100)],
        ... )

    The training throughput is logged by the :class:`~composer.loggers.logger.Logger` to the following keys as
    described below.

    +-----------------------+-------------------------------------------------------------+
    | Key                   | Logged data                                                 |
    +=======================+=============================================================+
    |                       | Rolling average (over ``window_size`` most recent           |
    | ``samples/step``      | batches) of the number of samples processed per second      |
    |                       |                                                             |
    +-----------------------+-------------------------------------------------------------+
    |                       | Number of samples processed per second (averaged over       |
    | ``samples/epoch``     | an entire epoch)                                            |
    +-----------------------+-------------------------------------------------------------+
    | ``wall_clock/train``  | Total elapsed training time                                 |
    +-----------------------+-------------------------------------------------------------+
    | ``wall_clock/val``    | Total elapsed validation time                               |
    +-----------------------+-------------------------------------------------------------+
    | ``wall_clock/total``  | Total elapsed time (wall_clock/train + wall_clock/val)      |
    +-----------------------+-------------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Default to 100.
    """

    def __init__(self, window_size: int = 100):
        # Track the epoch num samples and wct to compute throughput over the entire epoch
        self.epoch_start_num_samples = Time.from_sample(0)
        self.epoch_start_wct = datetime.timedelta(0)

        # Track the batch num samples and wct to compute throughput over a window of batches
        self.batch_start_num_samples = Time.from_sample(0)
        self.batch_start_wct = datetime.timedelta(0)
        self.rolling_batch_times_and_num_samples: Deque[Tuple[datetime.timedelta,
                                                              Time[int]]] = deque(maxlen=window_size)
        self.window_size = window_size

        # To optimize performance for the the batch wct calculations, keep a running sum over the entire deque
        self.running_wct = datetime.timedelta(0)
        self.running_num_samples = Time.from_sample(0)

        # Keep track of time spent evaluating
        self.total_eval_wct = datetime.timedelta(0)

        self._loaded_state: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch_start_num_samples": self.epoch_start_num_samples,
            "epoch_start_wct": self.epoch_start_wct,
            "batch_start_num_samples": self.batch_start_num_samples,
            "batch_start_wct": self.batch_start_wct,
            "rolling_batch_times_and_num_samples": self.rolling_batch_times_and_num_samples,
            "running_wct": self.running_wct,
            "running_num_samples": self.running_num_samples,
            "total_eval_wct": self.total_eval_wct,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._loaded_state = state

    def _load_state(self) -> None:
        if self._loaded_state is not None:
            self.epoch_start_num_samples = self._loaded_state["epoch_start_num_samples"]
            self.epoch_start_wct = self._loaded_state["epoch_start_wct"]
            self.batch_start_num_samples = self._loaded_state["batch_start_num_samples"]
            self.batch_start_wct = self._loaded_state["batch_start_wct"]
            self.rolling_batch_times_and_num_samples = deque(
                [x for x in self._loaded_state["rolling_batch_times_and_num_samples"]],
                maxlen=self.window_size,
            )
            self.running_wct = self._loaded_state["running_wct"]
            self.running_num_samples = self._loaded_state["running_num_samples"]
            self.total_eval_wct = self._loaded_state["total_eval_wct"]
            self._loaded_state = None

    def epoch_start(self, state: State, logger: Logger):
        del logger  # unused
        self._load_state()
        self.rolling_batch_times_and_num_samples.clear()
        self.epoch_start_wct = state.timestamp.total_wct
        self.epoch_start_num_samples = state.timestamp.sample

    def batch_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self._load_state()
        self.batch_start_wct = state.timestamp.total_wct
        self.batch_start_num_samples = state.timestamp.sample

    def batch_end(self, state: State, logger: Logger):
        new_num_samples = state.timestamp.sample
        batch_num_samples = new_num_samples - self.batch_start_num_samples
        self.running_num_samples += batch_num_samples
        batch_wct = state.timestamp.total_wct - self.batch_start_wct
        self.running_wct += batch_wct

        if len(self.rolling_batch_times_and_num_samples) == self.window_size:
            # the buffer is full. subtract out the oldest num samples and wct from the running tallies, before the new entry is added
            oldest_wct, oldest_num_samples = self.rolling_batch_times_and_num_samples.popleft()
            self.running_num_samples -= oldest_num_samples
            self.running_wct -= oldest_wct

            # Log the throughput
            throughput = self.running_num_samples / self.running_wct.total_seconds()
            logger.data_batch({'samples/step': throughput})

        # Add the new element
        self.rolling_batch_times_and_num_samples.append((batch_wct, batch_num_samples))

        # Log the time
        logger.data_batch({
            "wall_clock/train": state.timestamp.total_wct,
            "wall_clock/val": self.total_eval_wct,
            "wall_clock/total": state.timestamp.total_wct + self.total_eval_wct,
        })

    def eval_end(self, state: State, logger: Logger):
        del logger  # unused
        self.total_eval_wct += state.eval_timestamp.total_wct

    def epoch_end(self, state: State, logger: Logger):
        epoch_time_in_train = state.timestamp.total_wct - self.epoch_start_wct
        train_examples_per_epoch = int(state.timestamp.sample - self.epoch_start_num_samples)

        logger.data_epoch({
            "samples/epoch": train_examples_per_epoch / epoch_time_in_train.total_seconds(),
        })
