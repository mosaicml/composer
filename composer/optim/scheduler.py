# Copyright 2021 MosaicML. All Rights Reserved.

"""Framework for and implementations of stateless learning rate schedulers.

Stateless schedulers solve some of the problems associated with PyTorch's built-in schedulers provided in 
:mod:`torch.optim.lr_scheduler`. Those schedulers use internal state to keep track of the current time, which is
incremented every time their ``.step()`` method is called. In practice, this means that PyTorch's schedulers can only
interpret the current time (or training progress) as a single integer: the number of times ``.step()`` has been called.
PyTorch's schedulers were written under the assumption that this value would represent the current epoch. This requires
that ``.step()`` be called exactly once per epoch.

A critical problem with this approach is that it oversimplifies the notion of time. Time can be measured in multiple
other units besides epochs, such as samples, batches, and even tokens for NLP datasets. PyTorch's schedulers are unable
to recognize this multiplicity, since their understanding of time is limited to how much ``.step()`` has been called.

To offer a concrete example, PyTorch's :class:`~torch.optim.lr_scheduler.MultiStepLR` is configured by a ``milestones``
parameter whose type is a list of integers. Each of these milestones represents a time at which the learning rate should
change. Implicitly, these milestones are expected to be epoch indices.

So what happens if you want to change the learning rate not after an epoch, but after a batch, as is common in some NLP
training loads? Despite that PyTorch's schedulers weren't designed for this, it's possible to call ``.step()`` after
every batch, rather than after every epoch. Unfortunately, if you do this, you need to adjust all timewise parameters of
your schedulers, since their unit has now been implicitly changed from epochs to batches.

The primary design goal of the stateless schedulers provided in this module is to allow schedulers to reason about
explicit time units via Composer's :mod:`~composer.core.time` abstraction. This means that schedulers can be configured
using arbitrary but explicit time units.

See :class:`~.ComposerSchedulerFn` for more information on stateless schedulers.

Attributes:
    ComposerScheduler (:attr:`~composer.core.types.Scheduler` or :class:`~.ComposerSchedulerFn`): Union type for
        representing both PyTorch's built-in schedulers and also Composer's stateless schedulers.
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
    "ComposerSchedulerFn", "ComposerScheduler", "compile", "step_scheduler", "multi_step_scheduler", "constant_scheduler",
    "linear_scheduler", "exponential_scheduler", "cosine_annealing_scheduler",
    "cosine_annealing_warm_restarts_scheduler", "polynomial_scheduler", "multi_step_with_warmup_scheduler",
    "linear_with_warmup_scheduler", "cosine_annealing_with_warmup_scheduler"
]


class ComposerSchedulerFn(Protocol):
    """Specification for a stateless scheduler function.

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

        Returns:
            alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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
    """Converts stateless schedulers into a PyTorch scheduler object.
    
    While the resulting scheduler provides a ``.step()`` interface similar to other PyTorch schedulers, the scheduler is
    also given a bound reference to the current :class:`~composer.core.State`. This means that any internal state updated
    by ``.step()`` can be ignored, and the scheduler can instead simply use the bound state to recalculate the current
    learning rate.

    If this function is provided an object that is already a PyTorch scheduler, it just returns the scheduler.

    Args:
        scheduler (ComposerScheduler): A scheduler, represented either as a :attr:`~composer.core.types.Scheduler` or as
            a :class:`~composer.optim.scheduler.ComposerSchedulerFn`.
        state (State): The Composer Trainer's state.

    Returns:
        compiled_scheduler (Scheduler): The scheduler, in a form compatible with PyTorch scheduler interfaces.
    """

    if isinstance(scheduler, Scheduler):
        return scheduler

    optimizers = state.optimizers
    if len(optimizers) != 1:
        raise NotImplementedError("Providing stateless schedulers is unsupported with multiple optimizers.")
    optimizer = optimizers[0]

    def scheduler_fn(epoch: int) -> float:
        del epoch  # unused
        return scheduler(state)

    lambda_scheduler = LambdaLR(optimizer, scheduler_fn)

    return lambda_scheduler


def step_scheduler(state: State, *, ssr: float = 1.0, step_size: Union[str, Time], gamma: float = 0.1) -> float:
    r"""Decays the learning rate discretely at fixed intervals.

    Analogous to :class:`~torch.optim.lr_scheduler.StepLR`.

    Decays the learning rate by ``gamma`` periodically, with a frequency determined by ``step_size``.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        step_size (str or Time): The amount of time between changes to the learning rate.
        gamma (float): Multiplicative decay factor. Default = ``0.1``.

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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

    Analogous to :class:`~torch.optim.lr_scheduler.MultiStepLR`.

    Decays the learning rate by ``gamma`` whenever a time milestone in ``milestones`` is reached.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        milestones (list of str or Time): Times at which the learning rate should change.
        gamma (float) Multiplicative decay factor. Default = ``0.1``.

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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
                       alpha: float = 1.0,
                       t_max: Union[str, Time] = '1dur') -> float:
    r"""Maintains a fixed learning rate.

    Analagous to :class:`~torch.optim.lr_scheduler.ConstantLR`.

    The default settings for this scheduler simply maintain a learning rate factor of 1 for the entire training
    duration. However, both the factor and the duration of this scheduler can be configured.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        alpha (float): The learning rate multiplier to output while this scheduler is active. Default = ``1.0``.
        t_max (str or Time): The duration of this scheduler. Default = ``'1dur'``.

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
    """

    total_time = _convert_time(t_max, state, ssr=ssr)

    if state.timer < total_time:
        return alpha

    return 1.0


def linear_scheduler(state: State,
                     *,
                     ssr: float = 1.0,
                     alpha_i: float = 1.0,
                     alpha_f: float = 0.0,
                     t_max: Union[str, Time] = '1dur') -> float:
    r"""Adjusts the learning rate linearly.

    Analogous to :class:`~torch.optim.lr_scheduler.LinearLR`.

    Linearly adjusts the learning rate multiplier from ``alpha_i`` to ``alpha_f`` over ``t_max`` time.

    .. warning::
        Note that the defaults for this scheduler differ from the defaults for  :class:`~torch.optim.lr_scheduler.LinearLR`.
        The PyTorch scheduler, by default, linearly increases the learning rate multiplier from 1.0 / 3 to 1.0, whereas
        this implementation, by default, linearly decreases the multiplier from 1.0 to 0.0.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        alpha_i (float): Initial learning rate multiplier. Default = ``1.0``.
        alpha_f (float): Final learning rate multiplier. Default = ``0.0``.
        t_max (str or Time): The duration of this scheduler. Default = ``'1dur'``.

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
    """

    total_time = _convert_time(t_max, state, ssr=ssr)
    current_time = state.timer.get(total_time.unit)
    frac_of_total = min(1.0, (current_time / total_time).value)

    current_factor = alpha_i + frac_of_total * (alpha_f - alpha_i)

    return current_factor


def exponential_scheduler(state: State, *, ssr: float = 1.0, gamma: float) -> float:
    r"""Decays the learning rate exponentially.

    Analogous to :class:`~torch.optim.lr_scheduler.ExponentialLR`.

    Decays the learning rate multiplier by a factor of ``gamma`` every epoch.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        gamma (float): Multiplicative decay factor.

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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
                               alpha_f: float = 0.0):
    r"""Decays the learning rate according to the decreasing part of a cosine curve.

    Analogous to :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.

    Specifically, the learning rate multiplier :math:`\eta` can be expressed as
    :math:`\alpha(\tau) = \alpha_f + (1 - \alpha_f) \times \frac{1}{2}(1 + \cos(\pi \times \tau))`, where :math:`\tau`
    represents the fraction of time elapsed :math:`t / t_{max}` (clipped to the interval :math:`[0, 1]`), and
    :math:`\alpha_f` represents the learning rate multiplier to decay to.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        t_max (str or Time): Total time. Default = ``'1dur'``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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
                                             alpha_f: float = 0.0):
    r"""Cyclically decays the learning rate according to the decreasing part of a cosine curve.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
        t_0 (str or Time): The first cycle's duration.
        t_mult (float): The multiplier for subsequent cycles' durations. Default = ``1.0``.
        alpha_f (float): Minimum factor. Default = ``0.0``.

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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

    Returns:
        alpha (float): A multiplier to apply to the optimizer's provided learning rate.
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
