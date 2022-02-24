# Copyright 2021 MosaicML. All Rights Reserved.

import functools
from collections import Counter
from typing import Union

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, MultiStepLR, StepLR

from composer.core.types import Scheduler
from composer.optim.scheduler import ComposerSchedulerFn

__all__ = ["scale_scheduler"]


def _scale_pytorch_scheduler(scheduler: Scheduler, ssr: float):
    if isinstance(scheduler, StepLR):
        scheduler.step_size = int(scheduler.step_size * ssr)  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, MultiStepLR):
        milestones = scheduler.milestones  # type: ignore  -- unknown attribute
        milestones = Counter([int(ms * ssr) for ms in milestones])
        scheduler.milestones = milestones  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, CosineAnnealingLR):
        scheduler.T_max = int(scheduler.T_max * ssr)  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        # TODO: account for warmups
        scheduler.T_0 = int(scheduler.T_0 * ssr)  # type: ignore  -- unknown attribute
    elif isinstance(scheduler, ExponentialLR):
        factor = 1 / ssr
        scheduler.gamma = scheduler.gamma**factor  # type: ignore  -- unknown attribute
    else:
        raise ValueError(f'Scale schedule being applied to unrecognized Scheduler {scheduler}. '
                         'Please implement your scheduler as a function instead.')

    return scheduler


def _scale_composer_scheduler(scheduler: ComposerSchedulerFn, ssr: float):
    return functools.partial(scheduler, ssr=ssr)


def scale_scheduler(scheduler: Union[Scheduler, ComposerSchedulerFn],
                    ssr: float) -> Union[Scheduler, ComposerSchedulerFn]:
    """Makes a learning rate schedule take a different number of epochs.

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
    curve, at a learning rate near 0.

    .. doctest::

        >>> from composer.trainer._scale_schedule import scale_scheduler
        >>> from torch.optim.lr_scheduler import CosineAnnealingLR
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=90)
        >>> scheduler = scale_scheduler(scheduler, ssr=0.5)

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

    Raises:
        ValueError: If ``scheduler`` is not an instance of one of the above types.
    """
    if ssr <= 0:
        raise ValueError("Scale schedule ratio must be a positive value.")

    if isinstance(scheduler, Scheduler):
        return _scale_pytorch_scheduler(scheduler, ssr=ssr)
    elif callable(scheduler):
        return _scale_composer_scheduler(scheduler, ssr=ssr)
    else:
        raise ValueError(f'Received unknown scheduler {scheduler.__name__} of type {type(scheduler)}.')
