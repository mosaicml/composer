# Copyright 2021 MosaicML. All Rights Reserved.

# type: ignore

import logging
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, MultiStepLR, StepLR

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Scheduler
from composer.optim.scheduler import ConstantLR
from composer.utils import ensure_tuple

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
        scheduler.step_size = int(scheduler.step_size * ssr)
    elif isinstance(scheduler, MultiStepLR):
        scheduler.milestones = Counter([int(ms * ssr) for ms in scheduler.milestones])
    elif isinstance(scheduler, CosineAnnealingLR):
        assert orig_max_epochs is not None, "To scale Cosine decay, max_epochs must be provided."

        if hasattr(scheduler, 'interval') and scheduler.interval == "step":
            orig_max_epochs *= scheduler.steps_per_epoch

        warmup = orig_max_epochs - scheduler.T_max
        scheduler.T_max = int(orig_max_epochs * ssr - warmup)
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        scheduler.T_0 = int(scheduler.T_0 * ssr)  # TODO: account for warmups
    elif isinstance(scheduler, ExponentialLR):
        factor = 1 / ssr
        scheduler.gamma = scheduler.gamma**factor
    elif isinstance(scheduler, ConstantLR):
        return
    elif hasattr(scheduler, 'scale_schedule') and callable(scheduler.scale_schedule):
        scheduler.scale_schedule(ssr)
    else:
        raise ValueError(f'Scale schedule being applied to unrecognized Scheduler {scheduler}. '
                         'Please implement scale_schedule(ssr: float) method in your scheduler.')


@dataclass
class ScaleScheduleHparams(AlgorithmHparams):
    """See :class:`ScaleSchedule`"""

    ratio: float = hp.required('Ratio to scale the schedule.', template_default=1.0)
    method: str = hp.optional("Method to scale the schedule, one of 'epoch' or 'samples'. Default: epoch.",
                              default='epoch')

    def __post_init__(self):
        assert self.method in ('epoch', 'samples'), "Scale schedule method must be one of epoch or samples."

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
            makes the schedule take half as many epochs and 2.0 makes it
            take twice as many epochs.
        method: Currently only ``"epochs"`` is supported.

    Raises:
        ValueError: Raised during ``apply`` if ``scheduler`` is not supported by :func:`scale_scheduler`.
        ValueError: Raised during ``apply`` if the resulting number of epochs after scaling the
            learning rate schedule is zero.
        NotImplementedError: Raised during ``apply`` if ``method != "epochs"``.

    See also:
        :func:`scale_scheduler`
    """

    def __init__(self, ratio: float, method: str = 'epoch'):
        self.hparams = ScaleScheduleHparams(ratio=ratio, method=method)
        self.activated = False

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.TRAINING_START
        
        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run no
        """
        return event == Event.TRAINING_START

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Rescales the number of epochs spanned by `state`'s learning rate schedule.

        Raises:
            ValueError: If ``scheduler`` is not supported by :func:`scale_scheduler`.
            ValueError: If the resulting number of epochs after scaling the
                learning rate schedule is zero.
            NotImplementedError: If ``self.method == 'samples'``.
        """
        assert self.activated is False, "Scale Schedule should only be run once, check your control flow."
        assert state.schedulers is not None

        orig_max_epochs = state.max_epochs
        new_max_epochs = int(state.max_epochs * self.hparams.ratio)
        log.info(f'max_epochs changed from {state.max_epochs} to {new_max_epochs}')
        state.max_epochs = new_max_epochs
        if state.max_epochs == 0:
            raise ValueError('Scale schedule has reduced the max_epochs to 0. Set a higher ratio or more epochs.')

        if hasattr(state.schedulers, 'schedulers'):
            schedulers = state.schedulers.schedulers
        else:
            schedulers = ensure_tuple(state.schedulers)

        if self.hparams.method == 'epoch':
            for scheduler in schedulers:
                scale_scheduler(scheduler, self.hparams.ratio, orig_max_epochs)
        elif self.hparams.method == 'samples':
            raise NotImplementedError('Scale schedule algorithm with samples method not supported yet.')

        self.activated = True
