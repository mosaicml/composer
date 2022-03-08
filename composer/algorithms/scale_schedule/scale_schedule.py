# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import warnings
from typing import Optional

from composer.core import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)


class ScaleSchedule(Algorithm):
    """Deprecated - do not use.

    This algorithm is deprecated, and is being replaced by the scale_schedule_ratio param
    supported directly by the Composer Trainer. For backwards compatibility, the Composer
    Trainer detects when this algorithm has been initialized, and pulls the `ratio` param
    accordingly.

    Args:
        ratio (float, optional): The factor by which to scale the duration of the schedule. E.g., 0.5
            makes the schedule take half as long and 2.0 makes it
            take twice as long. Default: ``1.0``.
    """

    def __init__(self, ratio: float = 1.0):
        self.ratio = ratio
        warnings.warn(
            "ScaleScheduleDeprecationWarning: The scale schedule algorithm is deprecated. "
            "Please instead use the scale_schedule_ratio parameter of the Composer Trainer.",
            category=DeprecationWarning)

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.INIT.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run.
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """No-op."""
