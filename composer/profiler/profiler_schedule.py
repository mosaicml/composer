# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Profiler Schedules."""

from typing import Callable

from composer.core.state import State
from composer.profiler.profiler_action import ProfilerAction

__all__ = ['cyclic_schedule']


def cyclic_schedule(
    skip_first: int = 0,
    wait: int = 0,
    warmup: int = 1,
    active: int = 4,
    repeat: int = 1,
) -> Callable[[State], ProfilerAction]:
    """Profiler schedule function for a cyclic profiling window.

    This function returns a schedule function that uses a cyclic profiling window. The resulting function can be
    passed as the ``prof_schedule`` argument to the :class:`.Trainer`.

    The cyclic window skips the first ``skip_first`` + ``resumption_batch_idx`` batches in every epoch.
    ``resumption_batch_idx`` is accessed from state.profiler. It is the ``state.timestamp.batch_in_epoch``
    when resuming training.  Then, it performs a cycle of skipping ``wait`` batches, warming up for ``warmup``
    batches, and recording ``active`` batches. It repeats this cycle up to ``repeat`` times per epoch (or
    for the entire epoch, if ``repeat`` is 0). This logic repeats every epoch.

    Args:
        skip_first (int, optional): Number of batches to skip profiling at epoch start.  Defaults to ``0``.
        wait (int, optional): For each profiling cycle, number of batches to skip at the beginning of the cycle.
            Defaults to ``0``.
        warmup (int, optional): For each profiling cycle, number of batches to be in the warmup state after skipping
            ``wait`` batches. Defaults to ``1``.
        active (int, optional): For each profiling cycle, number of batches to record after warming up.  Defaults to ``4``.
        repeat (int, optional): Number of profiling cycles to perform per epoch. Set to ``0`` to record the entire epoch.
            Defaults to ``1``.

    Returns:
        (State -> ProfilerAction): A ``prof_schedule`` for the :class:`.Trainer`.
    """

    def schedule(state: State):
        # do wait, then warump, then active, up to repeat times per cycle
        cycle_len = wait + warmup + active
        batch_idx = int(state.timestamp.batch_in_epoch)
        if state.profiler is not None:
            skip_first_after_resumption = skip_first + state.profiler.resumption_batch_idx
        else:
            skip_first_after_resumption = skip_first
        if batch_idx < skip_first_after_resumption:
            return ProfilerAction.SKIP
        if repeat != 0 and batch_idx >= cycle_len * repeat + skip_first_after_resumption:
            # exhausted the repeat
            return ProfilerAction.SKIP
        position_in_cycle = (batch_idx - skip_first_after_resumption) % cycle_len
        if position_in_cycle < wait:
            return ProfilerAction.SKIP
        if position_in_cycle < wait + warmup:
            return ProfilerAction.WARMUP
        is_last_batch_in_epoch = state.dataloader_len is not None and state.timestamp.batch_in_epoch == state.dataloader_len - 1
        if position_in_cycle == cycle_len - 1 or is_last_batch_in_epoch:
            return ProfilerAction.ACTIVE_AND_SAVE
        return ProfilerAction.ACTIVE

    return schedule
