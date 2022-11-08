# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Action states for the :class:`Profiler` that define whether or not events are being recorded to the trace file."""

from composer.utils import StringEnum

__all__ = ['ProfilerAction']


class ProfilerAction(StringEnum):
    """Action states for the :class:`Profiler` that define whether or not events are being recorded to the trace file.

    Attributes:
        SKIP: Do not record new events to the trace.  Any events started during ``ACTIVE`` or ``WARMUP`` will be recorded upon finish.
        WARMUP: Record all events to the trace `except` those requiring a warmup period to initialize data structures (e.g., :doc:`profiler`).
        ACTIVE: Record all events to the trace.
        ACTIVE_AND_SAVE: Record all events and save the trace at the end of the batch.
    """
    SKIP = 'skip'
    WARMUP = 'warmup'
    ACTIVE = 'active'
    ACTIVE_AND_SAVE = 'active_and_save'
