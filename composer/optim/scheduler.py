# Copyright 2021 MosaicML. All Rights Reserved.

"""Framework for and implementations of stateless learning rate schedulers.

Stateless schedulers solve some of the problems associated with PyTorch's built-in schedulers provided in 
:mod:`torch.optim.lr_scheduler`. Those schedulers use internal state to keep track of the current time, which is
incremented every time their ``.step()`` method is called. In practice, this means that PyTorch's schedulers can only
interpret the current time (or training progress) as a single integer: the number of times ``.step()`` has been called.
PyTorch's schedulers were written under the assumption that this value would represent the current epoch. This requires
that ``.step()`` be called exactly once per epoch.

A critical problem with this approach is that it oversimplifies the notion of time. Time can be measured in multiple
other units besides epochs, such as samples, batches, and even tokens for NLP datasets, and in practice it can be useful
to use even a combination of these units in configuring schedulers.

A second problem is that there are major benefits to reap from updating a scheduler more frequently than just every
epoch. It is commonly found that updating a scheduler after every batch improves model accuracy. With PyTorch's

However, time can be measured in multiple other units, such as samples, batches, and even tokens for NLP datasets. 
It can be useful to represent scheduler parameters See :mod:`~composer.core.time` for more information on how the Composer library 
handles representations of time.

See `~.ComposerSchedulerFn` for the definition of a stateless scheduler.
"""

import logging
import math
import warnings
from typing import TYPE_CHECKING, List, Union

from torch.optim.lr_scheduler import LambdaLR

from composer.core import State
from composer.core.time import Time, TimeUnit
from composer.core.types import Scheduler

try:
    from typing import Protocol
except ImportError:
    Protocol = object  # Protocol is not available in python 3.7

if TYPE_CHECKING:
    from typing import Protocol

log = logging.getLogger(__name__)

__all__ = [
    "ComposerSchedulerFn", "compile", "step_scheduler", "multi_step_scheduler", "constant_scheduler",
    "linear_scheduler", "exponential_scheduler", "cosine_annealing_scheduler",
    "cosine_annealing_warm_restarts_scheduler", "polynomial_scheduler", "multi_step_with_warmup_scheduler",
    "linear_with_warmup_scheduler", "cosine_annealing_with_warmup_scheduler"
]


class ComposerSchedulerFn(Protocol):
    """Specification for a "stateless" scheduler function.

    A scheduler function should be a pure function that returns a multiplier to apply to the optimizer's provided
    learning rate, given the current trainer state, and optionally a "scale schedule ratio" (SSR). A typical
    implementation will read ``state.timer``, and possibly other fields like ``state.max_duration``, to determine the
    trainer's latest temporal progress.
    """

    def __call__(self, state: State, *, ssr: float = 1.0) -> float:
        """Calculate the current learning rate factor.

        Args:
            state (State): The current Composer Trainer state.
            ssr (float): The scale schedule ratio. In general, the learning rate computed by this
                scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
                this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        """
        raise NotImplementedError


ComposerScheduler = Union[Scheduler, ComposerSchedulerFn]


def _convert_time(time: Union[str, Time], state: State, ssr: float = 1.0) -> Time[int]:
    if isinstance(time, str):
        time = Time.from_timestring(time)

    if time.unit == TimeUnit.DURATION:
        assert state.max_duration.unit == TimeUnit.EPOCH  # Enforced by the trainer
        max_duration_batches = state.max_duration.value * state.steps_per_epoch
        return Time(value=int(time.value * max_duration_batches), unit=TimeUnit.BATCH)

    if time.unit == TimeUnit.EPOCH:
        time = Time(value=time.value * state.steps_per_epoch, unit=TimeUnit.BATCH)

    return Time(value=int(time.value * ssr), unit=time.unit)


def compile(scheduler: ComposerScheduler, state: State) -> Scheduler:

    if isinstance(scheduler, Scheduler):
        return scheduler

    optimizers = state.optimizers
    if len(optimizers) != 1:
        raise NotImplementedError("Providing functional schedulers is unsupported with multiple optimizers.")
    optimizer = optimizers[0]

    def scheduler_fn(epoch: int) -> float:
        del epoch  # unused
        return scheduler(state)

    lambda_scheduler = LambdaLR(optimizer, scheduler_fn)

    return lambda_scheduler


def step_scheduler(state: State, *, ssr: float = 1.0, step_size: Union[str, Time], gamma: float = 0.1) -> float:
    r"""Decays the learning rate discretely at fixed intervals.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        gamma (float): Gamma. Default = ``0.1``.
    """

    step_size = _convert_time(step_size, state, ssr=ssr)
    current_time = state.timer.get(step_size.unit)
    steps = int(current_time / step_size)

    return gamma**steps


def multi_step_scheduler(state: State,
                         *,
                         ssr: float = 1.0,
                         milestones: List[Union[str, Time]],
                         gamma: float = 0.1) -> float:
    r"""Decays the learning rate discretely at fixed milestones.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        milestones (list of str or Time): Milestones.
        gamma (float) Gamma. Default = ``0.1``.
    """

    milestones = [_convert_time(milestone, state, ssr=ssr) for milestone in milestones]

    factor = 1.0
    for milestone in milestones:
        if state.timer >= milestone:
            factor *= gamma

    return factor


def constant_scheduler(state: State,
                       *,
                       ssr: float = 1.0,
                       factor: float = 1.0,
                       total_time: Union[str, Time] = '1dur') -> float:
    r"""Maintains a fixed learning rate.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        factor (float): Factor. Default = ``1.0``.
        total_time (str or Time): Total time. Default = ``'1dur'``.
    """

    total_time = _convert_time(total_time, state, ssr=ssr)

    if state.timer < total_time:
        return factor

    return 1.0


def linear_scheduler(state: State,
                     *,
                     ssr: float = 1.0,
                     start_factor: float = 1.0,
                     end_factor: float = 0.0,
                     total_time: Union[str, Time] = '1dur') -> float:
    r"""Adjusts the learning rate linearly.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        start_factor (float): Start factor. Default = ``1.0``.
        end_factor (float): End factor. Default = ``0.0``.
        total_time (str or Time): Total time. Default = ``'1dur'``.
    """

    total_time = _convert_time(total_time, state, ssr=ssr)
    current_time = state.timer.get(total_time.unit)
    frac_of_total = min(1.0, (current_time / total_time).value)

    current_factor = start_factor + frac_of_total * (end_factor - start_factor)

    return current_factor


def exponential_scheduler(state: State, *, ssr: float = 1.0, gamma: float) -> float:
    r"""Decays the learning rate exponentially.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        gamma (float): Gamma.
    """

    current_time = state.timer.epoch

    return gamma**(current_time.value / ssr)


def _cosine_anneal(x: float, min_y: float = 0, max_y: float = 1) -> float:
    """Implements a cosine decay curve.

    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range [min_y, max_y]. Additionally, param x is
    clipped to the interval [0, 1].
    """

    x = min(max(x, 0.0), 1.0)
    return min_y + (max_y - min_y) * (1 + math.cos(x * math.pi)) / 2


def cosine_annealing_scheduler(state: State,
                               *,
                               ssr: float = 1.0,
                               t_max: Union[str, Time] = '1dur',
                               min_factor: float = 0.0):
    r"""Decays the learning rate according to the decreasing part of a cosine curve.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        t_max (str or Time): Total time. Default = ``'1dur'``.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    t_max = _convert_time(t_max, state, ssr=ssr)
    current_time = state.timer.get(t_max.unit)
    frac_of_total = (current_time / t_max).value

    return _cosine_anneal(x=frac_of_total, min_y=min_factor)


def cosine_annealing_warm_restarts_scheduler(state: State,
                                             *,
                                             ssr: float = 1.0,
                                             t_0: Union[str, Time],
                                             t_mult: float = 1.0,
                                             min_factor: float = 0.0):
    r"""Cyclically decays the learning rate according to the decreasing part of a cosine curve.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        t_0 (str or Time): The first cycle's duration.
        t_mult (float): The multiplier for subsequent cycles' durations. Default = ``1.0``.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    t_0 = _convert_time(t_0, state, ssr=ssr)
    current_interval_len = t_0
    current_interval_end = t_0
    while current_interval_end <= state.timer.get(current_interval_end.unit):
        if current_interval_len.value == 0:
            raise ValueError('Interval between restarts for cosine annealing/warm restarts scheduler has decayed to 0.')

        current_interval_len = Time(value=int(t_mult * current_interval_len.value), unit=current_interval_len.unit)
        current_interval_end += current_interval_len

    current_interval_start = current_interval_end - current_interval_len
    frac_of_current_interval = ((state.timer.get(t_0.unit) - current_interval_start) / current_interval_len).value

    return _cosine_anneal(x=frac_of_current_interval, min_y=min_factor)


def polynomial_scheduler(state: State,
                         *,
                         ssr: float = 1.0,
                         t_max: Union[str, Time] = '1dur',
                         power: float,
                         min_factor: float = 0.0):
    r"""Sets the learning rate to be exponentially proportional to the percentage of training time left.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        t_max (str or Time): Total time. Default = ``'1dur'``.
        power (float): Power.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    t_max = _convert_time(t_max, state, ssr=ssr)
    current_time = state.timer.get(t_max.unit)
    frac_of_total = (current_time / t_max).value

    coeff = (1 - frac_of_total)**power
    current_factor = min_factor + coeff * (1.0 - min_factor)
    return current_factor


def multi_step_with_warmup_scheduler(state: State,
                                     *,
                                     ssr: float = 1.0,
                                     warmup_time: Union[str, Time],
                                     milestones: List[Union[str, Time]],
                                     gamma: float = 0.1) -> float:
    r"""Decays the learning rate discretely at fixed milestones, with a linear warmup.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        warmup_time (str or Time): Warmup time.
        milestones (list of str or Time): Milestones.
        gamma (float) Gamma. Default = ``0.1``.
    """

    # N.B. warmup time is intentionally *not* subject to scale schedule
    warmup_time = _convert_time(warmup_time, state)
    if warmup_time.value == 0:
        warnings.warn("The warmup duration is 0. If you specified warmup as a fraction of total "
                      "training duration, take note that the warmup duration is calculated in the "
                      "same unit as the trainer's max_duration parameter.")

    if state.timer < warmup_time:
        return linear_scheduler(state, start_factor=0.0, end_factor=1.0, total_time=warmup_time)

    return multi_step_scheduler(state, ssr=ssr, milestones=milestones, gamma=gamma)


def linear_with_warmup_scheduler(state: State,
                                 *,
                                 ssr: float = 1.0,
                                 warmup_time: Union[str, Time],
                                 start_factor: float = 1.0,
                                 end_factor: float = 0.0,
                                 total_time: Union[str, Time] = '1dur'):
    r"""Adjusts the learning rate linearly, with a linear warmup.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        warmup_time (str or Time): Warmup time.
        start_factor (float): Start factor. Default = ``1.0``.
        end_factor (float): End factor. Default = ``0.0``.
        total_time (str or Time): Total time. Default = ``'1dur'``.
    """

    # N.B. warmup time is intentionally *not* subject to scale schedule
    warmup_time = _convert_time(warmup_time, state)
    if warmup_time.value == 0:
        warnings.warn("The warmup duration is 0. If you specified warmup as a fraction of total "
                      "training duration, take note that the warmup duration is calculated in the "
                      "same unit as the trainer's max_duration parameter.")

    if state.timer < warmup_time:
        return linear_scheduler(state, start_factor=0.0, end_factor=start_factor, total_time=warmup_time)

    total_time = _convert_time(total_time, state, ssr=ssr)
    current_time = state.timer.get(warmup_time.unit)
    frac_of_total = min(1.0, ((current_time - warmup_time) / (total_time - warmup_time)).value)

    current_factor = start_factor + frac_of_total * (end_factor - start_factor)

    return current_factor


def cosine_annealing_with_warmup_scheduler(state: State,
                                           *,
                                           ssr: float = 1.0,
                                           warmup_time: Union[str, Time],
                                           t_max: Union[str, Time] = '1dur',
                                           min_factor: float = 0.0):
    r"""Decays the learning rate according to the decreasing part of a cosine curve, with a linear warmup.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        warmup_time (str or Time): Warmup time.
        t_max (str or Time): Total time. Default = ``'1dur'``.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    # N.B. warmup time is intentionally *not* subject to scale schedule
    warmup_time = _convert_time(warmup_time, state)
    if warmup_time.value == 0:
        warnings.warn("The warmup duration is 0. If you specified warmup as a fraction of total "
                      "training duration, take note that the warmup duration is calculated in the "
                      "same unit as the trainer's max_duration parameter.")

    if state.timer < warmup_time:
        return linear_scheduler(state, start_factor=0.0, end_factor=1.0, total_time=warmup_time)

    t_max = _convert_time(t_max, state, ssr=ssr)
    current_time = state.timer.get(warmup_time.unit)
    frac_of_total = ((current_time - warmup_time) / (t_max - warmup_time)).value

    return _cosine_anneal(x=frac_of_total, min_y=min_factor)
