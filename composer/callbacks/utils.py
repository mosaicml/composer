# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Callable, Optional, Set, Union

from composer.core import Event, State, Time
from composer.utils.misc import create_interval_scheduler as _create_interval_scheduler


def create_interval_scheduler(interval: Union[str, int, Time],
                              include_end_of_training: bool = True,
                              checkpoint_events: bool = True,
                              final_events: Optional[Set[Event]] = None) -> Callable[[State, Event], bool]:
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
