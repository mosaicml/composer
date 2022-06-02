# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import time
import warnings
from typing import Sequence

from composer.core import State
from composer.core.callback import Callback
from composer.core.types import BreakEpochException
from composer.loggers import Logger
from composer.core.time import TimeUnit
from composer.core.time import ensure_time
from composer.core.time import Time

log = logging.getLogger(__name__)


class Benchmarker(Callback):
    """Fast-forward the training loop to record throughput for specific epochs and/or steps.
    It modifies the :attr:`~composer.core.state.State.step` and
    :attr:`~composer.core.state.State.epoch` to fast-forward the training loop,
    so that algorithms that activate at specific times will trigger
    and can be profiled.
    It stops the training process after the last step or epoch is profiled.
    It logs:
    * The throughput (averaged over the number of steps profiled in an epoch)
      to the ``throughput/step`` key.
    * The total elapsed training time to the ``wall_clock/train`` key.
    .. warning::
        The :class:`Benchmarker`: should NOT be used in conjunction with the
        :class:`~composer.callbacks.speed_monitor.SpeedMonitor`, since they
        log to the same keys.
    .. warning::
        The :class:`Benchmarker`: modifies the :class:`~composer.core.State`,
        which is an exception to the convention that callbacks should NOT
        modify state. This callback may break other algorithms and callbacks.
    Args:
        window_length (int, optional):
            Number of steps to profile at each entry of :attr:`step_list`.
            Defaults to 50.
        epoch_list (Sequence[int], optional).
            List of epochs at which to measure throughput.
            Defaults to [0, 1].
        step_list (Sequence[int], optional).
            List of steps at which to measure throughput.
            Defaults to [0, 50].
        all_epochs (bool, optional).
            Whether to override epoch_list and profile at all epochs.
            If False (the default), then it fast-forwards to
            the steps and epochs being profiled (specified by ``epoch_list``
            and ``step_list``, respectively).
            Otherwise, if True, then the throughput for
            the first ``window_length`` batches of every epoch are recorded.
    """

    def __init__(self,
                 window_length: int = 50,
                 epoch_list: Sequence[int] = (0, 1),
                 step_list: Sequence[int] = (0, 50),
                 all_epochs: bool = False):
        super().__init__()
        if not all_epochs:
            if len(epoch_list) == 0:
                raise ValueError("'epoch_list'  must be non-empty.")
            if 0 not in epoch_list:
                raise ValueError("'epoch_list' must contain 0, otherwise first K epochs have unknown speed")
        if len(step_list) == 0:
            raise ValueError("'step_list'  must be non-empty.")
        if 0 not in step_list:
            raise ValueError("'step_list' must contain 0 because `EPOCH_START` requires batch_idx of 0")

        # Sort lists so that time only moves forward
        epoch_list = list(sorted(epoch_list))
        step_list = list(sorted(step_list))

        self.current_time = None
        self.batch_start_num_samples = None
        self.profile_examples = 0
        self.profile_steps = 0
        self.profile_time = 0
        self.wall_clock_train = 0

        self.window_length = window_length

        self.all_epochs = all_epochs
        self.epoch_list = epoch_list
        self.epoch_ix = 0
        self.step_list = step_list
        self.step_ix = 0

        # initialized on the fit_start event
        self.original_max_duration = -1
        self.wct_dict = {}

    def _compute_elapsed_wct(self, epoch_wct_dict, steps_per_epoch: int, n_epochs: int):
        wct = 0.0
        wct_per_step = 0
        assert 0 in epoch_wct_dict, "epoch_wct_dict must contain 0"

        for step in range(int(steps_per_epoch)):
            if step in epoch_wct_dict:
                wct_per_step = epoch_wct_dict[step]
            wct += wct_per_step
        return wct * n_epochs

    def fit_start(self, state: State, logger: Logger):
        del logger  # Unused
        warnings.warn("The benchmarker is activated. The model will not be fully trained."
                      "All quality metrics for this run will be incorrect.")
        self.wall_clock_train = 0.0
        self.original_max_duration = state.max_duration
        # maybe override epoch_list
        if self.all_epochs:
            self.epoch_list = list(range(state.max_duration))
            log.info(f"all_epochs=True, overriding epoch_list to be every epoch from 0 to {state.max_duration}")
        self.wct_dict = {e: {s: -1.0 for s in self.step_list} for e in self.epoch_list}

    def epoch_end(self, state: State, logger: Logger):
        prev_epoch = self.epoch_list[self.epoch_ix]
        epoch_wct_dict = self.wct_dict[prev_epoch]
        self.epoch_ix += 1
        if self.epoch_ix < len(self.epoch_list):
            next_epoch = self.epoch_list[self.epoch_ix]
        else:
            next_epoch = self.original_max_duration

        state.timestamp._epoch = Time(int(next_epoch), TimeUnit.EPOCH) 
        state.timestamp._batch = Time(int(next_epoch * int(state.dataloader_len)), TimeUnit.BATCH) 
        n_epochs = next_epoch - prev_epoch

        self.wall_clock_train += float(self._compute_elapsed_wct(epoch_wct_dict, state.dataloader_len, n_epochs))
        logger.data_epoch({'wall_clock/train': self.wall_clock_train})

    def batch_start(self, state: State, logger: Logger):
        del logger  # Unused
        # Todo: update the progress_bar so that it is compatible with Benchmarker and remove this print.
        print(">>> Benchmarker Info: Sampling at epoch #{} - batch #{}".format(state.timestamp.epoch, state.timestamp.batch))
        if self.current_time is None:
            self.current_time = time.time()
            self.profile_examples = 0
            self.profile_steps = 0
            self.profile_time = 0.0
            self.batch_start_num_samples = state.timestamp.sample

    def batch_end(self, state: State, logger: Logger):
        if self.current_time is not None:
            now = time.time()
            elapsed = now - self.current_time
            self.current_time = now
            new_num_samples = state.timestamp.sample
            batch_num_samples = new_num_samples - self.batch_start_num_samples
            self.profile_examples += int(batch_num_samples)
            self.profile_steps += 1
            self.profile_time += elapsed

            if self.profile_steps >= self.window_length:
                avg_throughput = self.profile_examples / self.profile_time
                avg_time_per_step = self.profile_time / self.profile_steps
                profile_epoch = self.epoch_list[self.epoch_ix]
                profile_step = self.step_list[self.step_ix]
                self.wct_dict[profile_epoch][profile_step] = avg_time_per_step
                logger.data_batch({'throughput/step': avg_throughput})

                self.current_time = None
                self.step_ix += 1
                if self.step_ix == len(self.step_list):
                    self.step_ix = 0
                    raise BreakEpochException
                else:
                    # Todo: I avoided defining setters in Timestamp() as it can cause potential bugs for others. 
                    # Instead, I overwrite the private members (despite not being an ideal practice).
                    new_batch_value = int(state.timestamp.epoch) * int(state.dataloader_len) + self.step_list[self.step_ix]

                    state.timestamp._batch = Time(new_batch_value, TimeUnit.BATCH)
                    state.timestamp._batch_in_epoch = Time(int(self.step_list[self.step_ix]), TimeUnit.BATCH)
