from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

import yahp as hp

from composer import Logger, State
from composer.callbacks.callback_hparams import CallbackHparams
from composer.core.callback import Callback
from composer.core.types import BreakEpochException

log = logging.getLogger(__name__)


class TimingMonitor(Callback):

    def __init__(self, min_steps: int, epoch_list: List[int], step_list: List[int], all_epochs: bool):
        super().__init__()
        self.hparams = TimingMonitorHparams(min_steps=min_steps,
                                            epoch_list=epoch_list,
                                            step_list=step_list,
                                            all_epochs=all_epochs)

        if not all_epochs:
            assert len(epoch_list) > 0, "'epoch_list'  must be non-empty."
            assert 0 in epoch_list, \
                "'epoch_list' must contain 0, otherwise first K epochs have unknown speed"
        assert len(step_list) > 0, "'step_list'  must be non-empty."
        assert 0 in step_list, \
            "'step_list' must contain 0 because `EPOCH_START` requires batch_idx of 0"

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

    def _compute_elapsed_wct(self, epoch_wct_dict, steps_per_epoch, n_epochs):
        wct = 0.0
        wct_per_step = 0
        assert 0 in epoch_wct_dict, "epoch_wct_dict must contain 0"
        for step in range(steps_per_epoch):
            if step in epoch_wct_dict:
                wct_per_step = epoch_wct_dict[step]
            wct += wct_per_step
        return wct * n_epochs

    def training_start(self, state: State, logger: Logger):
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
            self.profile_examples += state.last_batch_size * state.world_size
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


@dataclass
class TimingMonitorHparams(CallbackHparams):
    min_steps: int = hp.optional(
        doc="minimum number of steps to use for measuring throughput",
        default=50,
    )
    epoch_list: List[int] = hp.optional(
        doc="list of epochs at which to measure throughput",
        default_factory=lambda: [0, 1],
    )
    step_list: List[int] = hp.optional(
        doc="list of steps at which to measure throughput",
        default_factory=lambda: [0, 50],
    )
    all_epochs: bool = hp.optional(
        doc="if true, override epoch_list and profile at all epochs.",
        default=False,
    )

    def initialize_object(self) -> TimingMonitor:
        return TimingMonitor(
            min_steps=self.min_steps,
            epoch_list=self.epoch_list,
            step_list=self.step_list,
            all_epochs=self.all_epochs,
        )
