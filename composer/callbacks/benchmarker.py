# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import time
import warnings
from typing import Sequence

from composer import Logger, State
from composer.callbacks.callback_hparams import BenchmarkerHparams
from composer.core.callback import Callback
from composer.core.types import BreakEpochException
from composer.utils import ddp

log = logging.getLogger(__name__)


class Benchmarker(Callback):
    """Fast-forward the training loop to record 
    throughput for specific epochs and/or steps.

    It modifies the :attr:`~composer.core.state.State.step` and
    :attr:`~composer.core.state.State.epoch` to fast-forward the training loop,
    so that algorithms that activate at specific times will trigger
    and can be profiled.

    It stops the training process after the last step or epoch is profiled.

    It logs:

    * The throughput (averaged over the number of steps profiled in an epoch)
      to the ``throughput/step`` key.
    * The total elapsed training time to the ``wall_clock_train`` key.

    .. warning::
        The :class:`Benchmarker`: should NOT be used in conjunction with the
        :class:`~composer.callbacks.speed_monitor.SpeedMonitor`, since they
        log to the same keys.

    .. warning::
        The :class:`Benchmarker`: modifies the :class:`~compose.core.State`,
        which is an exception to the convention that callbacks should NOT
        modify state. This callback may break other algorithms and callbacks.

    Args:
        min_steps (int, optional):
            Maximum number of steps to profile per epoch, regardless of the length of
            regardless of the length of :attr:`step_list`.
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
            the first ``min_steps`` batches of every epoch are recorded.
    """

    def __init__(self,
                 min_steps: int = 50,
                 epoch_list: Sequence[int] = (0, 1),
                 step_list: Sequence[int] = (0, 50),
                 all_epochs: bool = False):
        super().__init__()
        self.hparams = BenchmarkerHparams(min_steps=min_steps,
                                          epoch_list=list(epoch_list),
                                          step_list=list(step_list),
                                          all_epochs=all_epochs)
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
        self.profile_examples = 0
        self.profile_steps = 0
        self.profile_time = 0
        self.wall_clock_train = 0

        self.min_steps = min_steps

        self.all_epochs = all_epochs
        self.epoch_list = epoch_list
        self.epoch_ix = 0
        self.step_list = step_list
        self.step_ix = 0

        # initialized at training_start
        self.original_max_epochs = -1
        self.wct_dict = {}

    def _compute_elapsed_wct(self, epoch_wct_dict, steps_per_epoch: int, n_epochs: int):
        wct = 0.0
        wct_per_step = 0
        assert 0 in epoch_wct_dict, "epoch_wct_dict must contain 0"
        for step in range(steps_per_epoch):
            if step in epoch_wct_dict:
                wct_per_step = epoch_wct_dict[step]
            wct += wct_per_step
        return wct * n_epochs

    def training_start(self, state: State, logger: Logger):
        del logger  # Unused
        warnings.warn("The timing monitor is activated. The model will not be fully trained."
                      "All quality metrics for this run will be incorrect.")
        self.wall_clock_train = 0.0
        self.original_max_epochs = state.max_epochs
        # maybe override epoch_list
        if self.all_epochs:
            self.epoch_list = list(range(state.max_epochs))
            log.info(f"all_epochs=True, overriding epoch_list to be every epoch from 0 to {state.max_epochs}")
        self.wct_dict = {e: {s: -1.0 for s in self.step_list} for e in self.epoch_list}
        state.max_epochs = len(self.epoch_list)

    def epoch_end(self, state: State, logger: Logger):
        prev_epoch = self.epoch_list[self.epoch_ix]
        epoch_wct_dict = self.wct_dict[prev_epoch]
        self.epoch_ix += 1
        if self.epoch_ix < len(self.epoch_list):
            next_epoch = self.epoch_list[self.epoch_ix]
        else:
            next_epoch = self.original_max_epochs

        state.epoch = next_epoch - 1
        state.step = next_epoch * state.steps_per_epoch
        n_epochs = next_epoch - prev_epoch

        self.wall_clock_train += self._compute_elapsed_wct(epoch_wct_dict, state.steps_per_epoch, n_epochs)
        logger.metric_epoch({'wall_clock_train': self.wall_clock_train})

    def batch_start(self, state: State, logger: Logger):
        del state, logger  # Unused
        if self.current_time is None:
            self.current_time = time.time()
            self.profile_examples = 0
            self.profile_steps = 0
            self.profile_time = 0.0

    def batch_end(self, state: State, logger: Logger):
        if self.current_time is not None:
            now = time.time()
            elapsed = now - self.current_time
            self.current_time = now
            self.profile_examples += state.last_batch_size * ddp.get_world_size()
            self.profile_steps += 1
            self.profile_time += elapsed

            if self.profile_steps >= self.min_steps:
                avg_throughput = self.profile_examples / self.profile_time
                avg_time_per_step = self.profile_time / self.profile_steps
                profile_epoch = self.epoch_list[self.epoch_ix]
                profile_step = self.step_list[self.step_ix]
                self.wct_dict[profile_epoch][profile_step] = avg_time_per_step
                logger.metric_batch({'throughput/step': avg_throughput})

                self.current_time = None
                self.step_ix += 1
                if self.step_ix == len(self.step_list):
                    self.step_ix = 0
                    raise BreakEpochException
                else:
                    state.step = state.epoch * state.steps_per_epoch + self.step_list[self.step_ix]
