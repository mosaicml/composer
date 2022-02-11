# Copyright 2021 MosaicML. All Rights Reserved.

"""Monitor throughput during training."""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

from composer.core import Logger, State
from composer.core.callback import Callback
from composer.core.types import StateDict

__all__ = ["SpeedMonitor"]


class SpeedMonitor(Callback):
    """Logs the training throughput to various keys.

    +-----------------------+-------------------------------------------------------------+
    | Key                   | Logged data                                                 |
    +=======================+=============================================================+
    |                       | Rolling average (over ``window_size`` most recent           |
    | ``throughput/step``   | batches) of the number of samples processed per second      |
    |                       |                                                             |
    +-----------------------+-------------------------------------------------------------+
    |                       | Number of samples processed per second (averaged over       |
    | ``throughput/epoch``  | an entire epoch)                                            |
    +-----------------------+-------------------------------------------------------------+
    |``wall_clock_train``   | Total elapsed training time                                 |
    +-----------------------+-------------------------------------------------------------+

    Args:
        window_size (int):
            Number of batches to use for a rolling average of throughput.
    """

    def __init__(self, window_size: int):
        super().__init__()
        self.train_examples_per_epoch = 0
        self.wall_clock_train = 0.0
        self.epoch_start_time = 0.0
        self.batch_start_num_samples = None
        self.batch_end_times: Deque[float] = deque(maxlen=window_size + 1)  # rolling list of batch end times
        self.batch_num_samples: Deque[int] = deque(maxlen=window_size)  # rolling list of num samples in batch.
        self.window_size = window_size
        self.loaded_state: Optional[StateDict] = None

    def state_dict(self) -> StateDict:
        """Returns a dictionary representing the internal state of the SpeedMonitor object.

        The returned dictionary is pickle-able via :meth:`torch.save`.

        Returns:
            StateDict: The state of the SpeedMonitor object
        """
        current_time = time.time()
        return {
            "train_examples_per_epoch": self.train_examples_per_epoch,
            "wall_clock_train": self.wall_clock_train,
            "epoch_duration": current_time - self.epoch_start_time,
            "batch_durations": [current_time - x for x in self.batch_end_times],
            "batch_num_samples": self.batch_num_samples,
        }

    def load_state_dict(self, state: StateDict) -> None:
        """Restores the state of SpeedMonitor object.

        Args:
            state (StateDict): The state of the object,
                as previously returned by :meth:`.state_dict`
        """
        self.loaded_state = state

    def _load_state(self) -> None:
        current_time = time.time()
        if self.loaded_state is not None:
            self.train_examples_per_epoch = self.loaded_state["train_examples_per_epoch"]
            self.wall_clock_train = self.loaded_state["wall_clock_train"]
            self.epoch_start_time = current_time - self.loaded_state["epoch_duration"]
            self.batch_end_times = deque([current_time - x for x in self.loaded_state["batch_durations"]],
                                         maxlen=self.window_size + 1)
            self.batch_num_samples = self.loaded_state["batch_num_samples"]
            self.loaded_state = None

    def batch_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~composer.core.event.Event.BATCH_START` event.

        Bookkeeping for the number of samples at the start of a batch.

        Args:
            state (State): The :class:`~composer.core.state.State` object
                used during training.
            logger (Logger):
                The :class:`~composer.core.logging.logger.Logger` object.
        """
        del logger  # unused
        self._load_state()
        self.batch_start_num_samples = state.timer.sample

    def epoch_start(self, state: State, logger: Logger):
        """Called on the :attr:`~composer.core.event.Event.EPOCH_START` event.

        Set/Reset bookkeeping variables at the start of an epoch

        Args:
            state (State): The :class:`~composer.core.state.State` object
                used during training.
            logger (Logger):
                The :class:`~composer.core.logging.logger.Logger` object.
        """
        del state, logger  # unused
        self._load_state()
        self.epoch_start_time = time.time()
        self.batch_end_times.clear()
        self.batch_num_samples.clear()
        self.train_examples_per_epoch = 0

    def batch_end(self, state: State, logger: Logger):
        """Called on the :attr:`~composer.core.event.Event.BATCH_END` event.

        If we have reached ``window_size`` batches, logs throughput to the ``throughput/step`` key.

        Args:
            state (State): The :class:`~composer.core.state.State` object
                used during training.
            logger (Logger):
                The :class:`~composer.core.logging.logger.Logger` object.
        """
        self.batch_end_times.append(time.time())
        new_num_samples = state.timer.sample
        batch_num_samples = int(new_num_samples - self.batch_start_num_samples)
        self.batch_num_samples.append(batch_num_samples)
        self.train_examples_per_epoch += batch_num_samples
        if len(self.batch_end_times) == self.window_size + 1:
            throughput = sum(self.batch_num_samples) / (self.batch_end_times[-1] - self.batch_end_times[0])
            logger.metric_batch({'throughput/step': throughput})

    def epoch_end(self, state: State, logger: Logger):
        """Called on the :attr:`~composer.core.event.Event.EPOCH_END` event.

        logs per epoch average throughput to the key ``throughput/epoch`` and wall clock train time to the
        ``wall_clock_train`` key.

        Args:
            state (State): The :class:`~composer.core.state.State` object
                used during training.
            logger (Logger):
                The :class:`~composer.core.logging.logger.Logger` object.
        """
        del state  # unused
        epoch_time = time.time() - self.epoch_start_time
        self.wall_clock_train += epoch_time
        logger.metric_epoch({
            "wall_clock_train": self.wall_clock_train,
        })
        logger.metric_epoch({
            "throughput/epoch": self.train_examples_per_epoch / epoch_time,
        })
