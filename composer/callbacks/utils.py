# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback utils."""

import warnings
from typing import Callable, Optional, Set, Union

from composer.core import Event, State, Time
from composer.utils.misc import create_interval_scheduler as _create_interval_scheduler


def create_interval_scheduler(interval: Union[str, int, Time],
                              include_end_of_training: bool = True,
                              checkpoint_events: bool = True,
                              final_events: Optional[Set[Event]] = None) -> Callable[[State, Event], bool]:
    """Helper function to create a scheduler according to a specified interval.

    Args:
        interval (Union[str, int, :class:`.Time`]): If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        include_end_of_training (bool): If true, the returned callable will return true at the end of training as well.
            Otherwise, the returned callable will return true at intervals only.
        checkpoint_events (bool): If true, will use the EPOCH_CHECKPOINT and BATCH_CHECKPOINT events. If False, will use
            the EPOCH_END and BATCH_END events.
        final_events (Optional[Set[Event]]): The set of events to trigger on at the end of training.

    Returns:
        Callable[[State, Event], bool]: A function that returns true at interval and at the end of training if specified.
            For example, it can be passed as the ``save_interval`` argument into the :class:`.CheckpointSaver`.
    """
    warnings.warn(
        '`composer.callbacks.utils.create_interval_scheduler has been moved to `composer.utils.misc.create_interval_scheduler` '
        + 'and will be removed in a future release.',
        DeprecationWarning,
    )
    return _create_interval_scheduler(
        interval=interval,
        include_end_of_training=include_end_of_training,
        checkpoint_events=checkpoint_events,
        final_events=final_events,
    )
