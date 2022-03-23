# Copyright 2021 MosaicML. All Rights Reserved.

from composer.utils.string_enum import StringEnum

__all__ = ["ProfilerAction"]


class ProfilerAction(StringEnum):
    """Action states for the :class:`Profiler` that define whether or not events are being recorded to the trace file.

    Attributes:
        SKIP: Do not record new events to the trace.  Any events started during ``ACTIVE`` or ``WARMUP`` will be recorded upon finish.
        WARMUP: Record all events to the trace `except` those requiring a warmup period to initialize data structures (e.g., :doc:`profiler`).
        ACTIVE: Record all events to the trace.
        ACTIVE_AND_SAVE: Record all events and save the trace, as the next action state will be :attr:`SKIP` or :attr:`WARMUP`
    """
    SKIP = "skip"
    WARMUP = "warmup"
    ACTIVE = "active"
    ACTIVE_AND_SAVE = "active_and_save"
