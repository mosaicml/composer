# Copyright 2021 MosaicML. All Rights Reserved.

from composer.utils.string_enum import StringEnum


class ProfilerAction(StringEnum):
    """Action states for the :class:`Profiler`.

    Attributes:
        SKIP: Not currently recording new events at the batch level or below.
            However, any open duration events will still be closed.
        WARMUP: The profiler
        ACTIVE: Record all events.
    """
    SKIP = "skip"
    WARMUP = "warmup"
    ACTIVE = "active"
