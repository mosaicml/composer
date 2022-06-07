# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

import datetime
from collections import deque
from typing import Any, Deque, Dict

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
        self.batch_wct_buffer: Deque[datetime.timedelta] = deque(maxlen=window_size)
        self.batch_num_samples_buffer: Deque[Time[int]] = deque(maxlen=window_size)
        self.window_size = window_size

        # To optimize performance for the the batch wct calculations, keep a running sum over the entire deque
        self.window_wct = datetime.timedelta(0)
        self.window_num_samples = Time.from_sample(0)

        # Keep track of time spent evaluating
        self.total_eval_wct = datetime.timedelta(0)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch_start_num_samples": self.epoch_start_num_samples,
            "epoch_start_wct": self.epoch_start_wct,
            "batch_start_num_samples": self.batch_start_num_samples,
            "batch_start_wct": self.batch_start_wct,
            "batch_wct_buffer": self.batch_wct_buffer,
            "batch_num_samples_buffer": self.batch_num_samples_buffer,
            "window_wct": self.window_wct,
            "window_num_samples": self.window_num_samples,
            "total_eval_wct": self.total_eval_wct,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.epoch_start_num_samples = state["epoch_start_num_samples"]
        self.epoch_start_wct = state["epoch_start_wct"]
        self.batch_start_num_samples = state["batch_start_num_samples"]
        self.batch_start_wct = state["batch_start_wct"]
        self.batch_wct_buffer = deque(
            [x for x in state["batch_wct_buffer"]],
            maxlen=self.window_size,
        )
        self.batch_num_samples_buffer = deque(
            [x for x in state["batch_num_samples_buffer"]],
            maxlen=self.window_size,
        )
        self.window_wct = state["window_wct"]
        self.window_num_samples = state["window_num_samples"]
        self.total_eval_wct = state["total_eval_wct"]

    def epoch_start(self, state: State, logger: Logger):
        del logger  # unused
        self.epoch_start_wct = state.timestamp.total_wct
        self.epoch_start_num_samples = state.timestamp.sample

    def batch_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_wct = state.timestamp.total_wct
        self.batch_start_num_samples = state.timestamp.sample

    def batch_end(self, state: State, logger: Logger):
        batch_num_samples = state.timestamp.sample - self.batch_start_num_samples
        self.window_num_samples += batch_num_samples
        batch_wct = state.timestamp.total_wct - self.batch_start_wct
        self.window_wct += batch_wct

        if len(self.batch_wct_buffer) == self.window_size:
            # the buffer is full. subtract out the oldest num samples and wct from the running tallies, before the new entry is added
            oldest_wct = self.batch_wct_buffer.popleft()
            oldest_num_samples = self.batch_num_samples_buffer.popleft()
            self.window_num_samples -= oldest_num_samples
            self.window_wct -= oldest_wct

            # Log the throughput
            throughput = int(self.window_num_samples) / self.window_wct.total_seconds()
            logger.data_batch({'samples/step': throughput})

        # Add the new element
        self.batch_wct_buffer.append(batch_wct)
        self.batch_num_samples_buffer.append(batch_num_samples)

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.data_batch({
            "wall_clock/train": state.timestamp.total_wct.total_seconds(),
            "wall_clock/val": self.total_eval_wct.total_seconds(),
            "wall_clock/total": (state.timestamp.total_wct + self.total_eval_wct).total_seconds(),
        })

    def eval_end(self, state: State, logger: Logger):
        del logger  # unused
        self.total_eval_wct += state.eval_timestamp.total_wct

    def epoch_end(self, state: State, logger: Logger):
        # `state.timestamp` excludes any time spent in evaluation
        epoch_time_in_train = state.timestamp.total_wct - self.epoch_start_wct
        train_examples_per_epoch = int(state.timestamp.sample - self.epoch_start_num_samples)

        logger.data_epoch({
            "samples/epoch": train_examples_per_epoch / epoch_time_in_train.total_seconds(),
        })
