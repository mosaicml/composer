# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import weakref
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional, cast

import yahp as hp
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, MultiStepLR, StepLR

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.core.time import Time
from composer.core.types import Scheduler
from composer.optim.scheduler import ConstantLR

log = logging.getLogger(__name__)


def scale_scheduler(scheduler: Scheduler, ssr: float, orig_max_epochs: Optional[int] = None):
    """Makes a learning rate schedule take a different number of epochs.

    See :class:`ScaleSchedule` for more information.

    Args:
        scheduler: A learning rate schedule object. Must be one of:

            * ``torch.optim.lr_scheduler.CosineAnnealingLR``
            * ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts``
            * ``torch.optim.lr_scheduler.ExponentialLR``
            * ``torch.optim.lr_scheduler.MultiStepLR``
            * ``torch.optim.lr_scheduler.StepLR``

        ssr: the factor by which to scale the duration of the schedule. E.g., 0.5
            makes the schedule take half as many epochs and 2.0 makes it
            take twice as many epochs.
        orig_max_epochs: the current number of epochs spanned by ``scheduler``.
            Used along with ``ssr`` to determine the new number of epochs
            ``scheduler`` should span.

    Raises:
        ValueError: If ``scheduler`` is not an instance of one of the above types.
    """
    if isinstance(scheduler, StepLR):
        scheduler.step_size = int(scheduler.step_size * ssr)  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, MultiStepLR):
        milestones = scheduler.milestones  # type: ignore  -- unknown attribute
        milestones = Counter([int(ms * ssr) for ms in milestones])
        scheduler.milestones = milestones  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, CosineAnnealingLR):
        assert orig_max_epochs is not None, "To scale Cosine decay, max_epochs must be provided."

        if hasattr(scheduler, 'interval') and scheduler.interval == "step":  # type: ignore  -- unknown attribute
            orig_max_epochs *= scheduler.steps_per_epoch  # type: ignore  -- unknown attribute

        warmup = orig_max_epochs - scheduler.T_max  # type: ignore  -- unknown attribute
        scheduler.T_max = int(orig_max_epochs * ssr - warmup)  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        # TODO: account for warmups
        scheduler.T_0 = int(scheduler.T_0 * ssr)  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, ExponentialLR):
        factor = 1 / ssr
        scheduler.gamma = scheduler.gamma**factor  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, ConstantLR):
        return
    elif hasattr(scheduler, 'scale_schedule') and callable(
            scheduler.scale_schedule):  # type: ignore  -- unknown attribute
        scheduler.scale_schedule(ssr)  # type: ignore  -- unknown attribute
    else:
        raise ValueError(f'Scale schedule being applied to unrecognized Scheduler {scheduler}. '
                         'Please implement scale_schedule(ssr: float) method in your scheduler.')


@dataclass
class ScaleScheduleHparams(AlgorithmHparams):
    """See :class:`ScaleSchedule`"""

    ratio: float = hp.required('Ratio to scale the schedule.', template_default=1.0)

    def initialize_object(self) -> "ScaleSchedule":
        return ScaleSchedule(**asdict(self))


class ScaleSchedule(Algorithm):
    """Makes the learning rate schedule take a different number of epochs.

    Training for less time is a strong baseline approach to speeding up
    training, provided that the training still gets through the entire
    learning rate schedule. E.g., training for half as long often yields
    little accuracy degredation, provided that the learning rate schedule
    is rescaled to take half as long as well. In contrast, if the schedule
    is not rescaled, training for half as long would mean simply stopping
    halfway through the training curve, which does reach nearly as
    high an accuracy.

    To see the difference, consider training for half as long using a cosine
    annealing learning rate schedule. If the schedule is not rescaled,
    training ends while the learning rate is still ~0.5. If the schedule is
    rescaled, training ends after passing through the full cosine
    curve, at a learning rate orders of .01 or smaller.

    Args:
        ratio: The factor by which to scale the duration of the schedule. E.g., 0.5
            makes the schedule take half as long and 2.0 makes it
            take twice as long.

    Raises:
        ValueError: Raised during ``apply`` if ``scheduler`` is not supported by :func:`scale_scheduler`.
        ValueError: Raised during ``apply`` if the resulting number of epochs after scaling the
            learning rate schedule is zero.

    See also:
        :func:`scale_scheduler`
    """

    def __init__(self, ratio: float):
        self.ratio = ratio

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.INIT.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run no
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Rescales the number of epochs spanned by `state`'s learning rate schedule.

        Raises:
            ValueError: If ``scheduler`` is not supported by :func:`scale_scheduler`.
            ValueError: If the resulting number of epochs after scaling the
                learning rate schedule is zero.
            NotImplementedError: If ``self.method == 'samples'``.
        """

        orig_max_duration = state.max_duration
        orig_max_epochs = state.max_epochs
        state.max_duration = cast(Time[int], orig_max_duration * self.ratio)
        log.info(f'max_duration changed from {orig_max_duration} to {state.max_duration}')
        if int(state.max_duration) == 0:
            raise ValueError('Scale schedule has reduced the max_duration to 0. Set a higher ratio or more epochs.')

        schedulers = []
        for scheduler in state.schedulers:
            if hasattr(scheduler, 'schedulers'):
                schedulers.extend(getattr(scheduler, "schedulers"))
            else:
                schedulers.append(scheduler)

        for scheduler in schedulers:
            scale_scheduler(scheduler, self.ratio, orig_max_epochs)
