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

See :class:`~.ComposerScheduler` for more information on stateless schedulers.
"""

import inspect
import logging
import math
import textwrap
import warnings
from typing import TYPE_CHECKING, List, Union

from torch.optim.lr_scheduler import LambdaLR

from composer.core import State
from composer.core.time import Time, TimeUnit

try:
    from typing import Protocol
except ImportError:
    Protocol = object  # Protocol is not available in python 3.7

if TYPE_CHECKING:
    from typing import Protocol

log = logging.getLogger(__name__)

__all__ = [
    "ComposerScheduler", "compile_composer_scheduler", "StepScheduler", "MultiStepScheduler", "ConstantScheduler",
    "LinearScheduler", "ExponentialScheduler", "CosineAnnealingScheduler", "CosineAnnealingWarmRestartsScheduler",
    "PolynomialScheduler", "MultiStepWithWarmupScheduler", "LinearWithWarmupScheduler",
    "CosineAnnealingWithWarmupScheduler"
]


class ComposerScheduler(Protocol):
    """Specification for a "stateless" scheduler function.

    A scheduler function should be a pure function that returns a multiplier to apply to the optimizer's provided
    learning rate, given the current trainer state, and optionally a "scale schedule ratio" (SSR). A typical
    implementation will read ``state.timer``, and possibly other fields like ``state.max_duration``, to determine the
    trainer's latest temporal progress.
    """

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        r"""Calculate the current learning rate factor.

        Args:
            state (State): The current Composer Trainer state.
            ssr (float): The scale schedule ratio. In general, the learning rate computed by this
                scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
                this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.

        Returns:
            alpha (float): A multiplier to apply to the optimizer's provided learning rate.
        """
        raise NotImplementedError


def _convert_time(time: Union[str, Time[int], Time[float]], state: State, ssr: float = 1.0) -> Time[int]:
    if isinstance(time, str):
        time = Time.from_timestring(time)

    if time.unit == TimeUnit.DURATION:
        if state.max_duration.unit == TimeUnit.EPOCH:
            return Time(int(time.value * state.steps_per_epoch * state.max_duration.value), TimeUnit.BATCH)
        return Time(int(time.value * state.max_duration.value), state.max_duration.unit)

    if time.unit == TimeUnit.EPOCH:
        # Epochs do not provide sufficient granularity for SSR scaling
        # e.g. if max_duration = 1ep, then any SSR would result in a new duration of 0.
        # so, convert the time into batches
        time = Time(value=time.value * state.steps_per_epoch, unit=TimeUnit.BATCH)

    return Time(value=int(time.value * ssr), unit=time.unit)


def compile_composer_scheduler(scheduler: ComposerScheduler, state: State, ssr: float = 1.0) -> LambdaLR:
    """Converts a stateless scheduler into a PyTorch scheduler object.

    While the resulting scheduler provides a ``.step()`` interface similar to other PyTorch schedulers, the scheduler is
    also given a bound reference to the current :class:`~composer.core.State`. This means that any internal state updated
    by ``.step()`` can be ignored, and the scheduler can instead simply use the bound state to recalculate the current
    learning rate.

    Args:
        scheduler (ComposerScheduler): A stateless scheduler, provided as a :class:`~.ComposerScheduler` object.
        state (State): The Composer Trainer's state.

    Returns:
        compiled_scheduler (Scheduler): The scheduler, in a form compatible with PyTorch scheduler interfaces.
    """

    optimizers = state.optimizers
    if len(optimizers) != 1:
        raise NotImplementedError("Providing functional schedulers is unsupported with multiple optimizers.")
    optimizer = optimizers[0]

    scheduler_sig = inspect.signature(scheduler)

    def scheduler_fn(epoch: int) -> float:
        del epoch  # unused. Provided by the pytorch LambdaLR

        # if the ssr is 1.0, don't pass it to the scheduler. This allows users to pass in lambdas that only take
        # one parameter -- the state
        if len(scheduler_sig.parameters) == 1:
            if ssr == 1.0:
                return scheduler(state)
            else:
                raise ValueError(
                    textwrap.dedent(f"""\
                    Scheduler {scheduler} does not support `scale_schedule_ratio`.
                    To use `scale_schedule_ratio`, the scheduler must take two arguments (state, ssr)"""))
        return scheduler(state, ssr)

    lambda_scheduler = LambdaLR(optimizer, scheduler_fn)

    return lambda_scheduler


class StepScheduler(ComposerScheduler):
    r"""Decays the learning rate discretely at fixed intervals.

    Analogous to :class:`~torch.optim.lr_scheduler.StepLR`.

    Decays the learning rate by a factor of ``gamma`` periodically, with a frequency determined by ``step_size``.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \gamma ^ {\text{floor}(t / \rho)}

    Where :math:`\rho` represents the time between changes to the learning rate (the step size), and
    :math:`\gamma` represents the multiplicative decay factor.
    
    Args:
        step_size (str or Time): Time between changes to the learning rate.
        gamma (float): Multiplicative decay factor. Default = ``0.1``.
    """

    def __init__(self, step_size: Union[str, Time], gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, state: State, ssr: float = 1.0):
        step_size = _convert_time(self.step_size, state, ssr=ssr)
        current_time = state.timer.get(step_size.unit)
        steps = int(current_time / step_size)

        return self.gamma**steps


class MultiStepScheduler(ComposerScheduler):
    r"""Decays the learning rate discretely at fixed milestones.

    Analogous to :class:`~torch.optim.lr_scheduler.MultiStepLR`.

    Decays the learning rate by a factor of ``gamma`` whenever a time milestone in ``milestones`` is reached.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \gamma ^ x

    Where :math:`x` represents the amount of milestones that have been reached, and :math:`\gamma` represents the
    multiplicative decay factor.
    
    Args:
        milestones (list of str or Time): Times at which the learning rate should change.
        gamma (float): Multiplicative decay factor. Default = ``0.1``.
    """

    def __init__(self, milestones: List[Union[str, Time]], gamma: float = 0.1):
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, state: State, ssr: float = 1.0):
        milestones = [_convert_time(milestone, state, ssr=ssr) for milestone in self.milestones]

        factor = 1.0
        for milestone in milestones:
            if state.timer >= milestone:
                factor *= self.gamma

        return factor


class ConstantScheduler(ComposerScheduler):
    r"""Maintains a fixed learning rate.

    Analagous to :class:`~torch.optim.lr_scheduler.ConstantLR`.

    The default settings for this scheduler simply maintain a learning rate factor of 1 for the entire training
    duration. However, both the factor and the duration of this scheduler can be configured.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases} \alpha, & \text{if } t < t_{max} \\ 1.0 & \text{otherwise} \end{cases}

    Where :math:`\alpha` represents the learning rate multiplier to maintain while this scheduler is active, and
    :math:`t_{max}` represents the duration of this scheduler.
    
    Args:
        alpha (float): Learning rate multiplier to maintain while this scheduler is active. Default = ``1.0``.
        t_max (str or Time): Duration of this scheduler. Default = ``'1dur'``.
    """

    def __init__(self, alpha: float = 1.0, t_max: Union[str, Time] = '1dur') -> None:
        self.alpha = alpha
        self.t_max = t_max

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        if state.timer < t_max:
            return self.alpha

        return 1.0


class LinearScheduler(ComposerScheduler):
    r"""Adjusts the learning rate linearly.

    Analogous to :class:`~torch.optim.lr_scheduler.LinearLR`.

    Linearly adjusts the learning rate multiplier from ``alpha_i`` to ``alpha_f`` over ``t_max`` time.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_i + (alpha_f - \alpha_i) \times \tau

    Given :math:`\tau`, the fraction of time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau = t / t_{max}
    
    Where :math:`\alpha_i` represents the initial learning rate multiplier, :math:`\alpha_f` represents
    the learning rate multiplier to decay to, and :math:`t_{max}` represents the duration of this scheduler.

    .. warning::
        Note that the defaults for this scheduler differ from the defaults for  :class:`~torch.optim.lr_scheduler.LinearLR`.
        The PyTorch scheduler, by default, linearly increases the learning rate multiplier from 1.0 / 3 to 1.0, whereas
        this implementation, by default, linearly decreases the multiplier from 1.0 to 0.0.
    
    Args:
        alpha_i (float): Initial learning rate multiplier. Default = ``1.0``.
        alpha_f (float): Final learning rate multiplier. Default = ``0.0``.
        t_max (str or Time): The duration of this scheduler. Default = ``'1dur'``.
    """

    def __init__(self, alpha_i: float = 1.0, alpha_f: float = 0.0, t_max: Union[str, Time] = '1dur'):
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_max = Time.from_timestring(t_max) if isinstance(t_max, str) else t_max

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(t_max.unit)
        frac_of_total = min(1.0, (current_time / t_max).value)

        current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)

        return current_factor


class ExponentialScheduler(ComposerScheduler):
    r"""Decays the learning rate exponentially.

    Analogous to :class:`~torch.optim.lr_scheduler.ExponentialLR`.

    Exponentially decays the learning rate such that it decays by a factor of ``gamma`` every ``decay_period`` time.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \gamma ^ {t / \rho}

    Where :math:`\rho` represents the decay period, and :math:`\gamma` represents the multiplicative decay factor.
    
    Args:
        decay_period (str or Time): Decay period. Default = ``'1ep'``.
        gamma (float): Multiplicative decay factor.
    """

    def __init__(self, gamma: float, decay_period: Union[str, Time] = '1ep'):
        self.gamma = gamma
        self.decay_period = decay_period

    def __call__(self, state: State, ssr: float = 1.0):
        decay_period = _convert_time(self.decay_period, state, ssr)
        current_time_in_decay_units = state.timer.get(decay_period.unit)

        return self.gamma**float(current_time_in_decay_units / decay_period)


def _cosine_anneal(x: float, min_y: float = 0.0, max_y: float = 1.0) -> float:
    """Implements a cosine decay curve.

    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range [min_y, max_y]. Additionally, param x is
    clipped to the interval [0, 1]
    """

    x = min(max(x, 0.0), 1.0)
    return min_y + (max_y - min_y) * (1 + math.cos(x * math.pi)) / 2


class CosineAnnealingScheduler(ComposerScheduler):
    r"""Decays the learning rate according to the decreasing part of a cosine curve.

    Analogous to :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_f + (1 - \alpha_f) \times \frac{1}{2} (1 + \cos(\pi \times \tau))

    Given :math:`\tau`, the fraction of time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau = t / t_{max}
    
    Where :math:`t_{max}`
    represents the duration of this scheduler, and :math:`\alpha_f` represents the learning rate multiplier to decay to.
    
    Args:
        t_max (str or Time): The duration of this scheduler. Default = ``'1dur'``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
    """

    def __init__(self, t_max: Union[str, Time] = '1dur', alpha_f: float = 0.0):
        self.t_max = t_max
        self.alpha_f = alpha_f

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(t_max.unit)
        frac_of_total = (current_time / t_max).value

        return _cosine_anneal(x=frac_of_total, min_y=self.alpha_f)


class CosineAnnealingWarmRestartsScheduler(ComposerScheduler):
    r"""Cyclically decays the learning rate according to the decreasing part of a cosine curve.

    Analogous to :class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.

    This scheduler resembles a regular cosine annealing curve, as seen in :class:`~.CosineAnnealingScheduler`, except
    that after the curve first completes ``t_0`` time, the curve resets to the start. The durations of subsequent cycles
    are each multiplied by ``t_mult``.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_f + (1 - \alpha_f) \times \frac{1}{2}(1 + \cos(\pi \times \tau_i))

    Given :math:`\tau_i`, the fraction of time elapsed through the :math:`i^\text{th}` cycle, as:

    .. math::
        \tau_i = (t - \sum_{j=0}^{i-1} t_0 t_{mult}^j) / (t_0 t_{mult}^i)
    
    Where :math:`t_0`
    represents the period of the first cycle, :math:`t_{mult}` represents the multiplier for the duration of successive
    cycles, and :math:`\alpha_f` represents the learning rate multiplier to decay to.
    
    Args:
        t_0 (str or Time): The period of the first cycle.
        t_mult (float): The multiplier for the duration of successive cycles. Default = ``1.0``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
    """

    def __init__(self, t_0: Union[str, Time], t_mult: float = 1.0, alpha_f: float = 0.0):
        self.t_0 = t_0
        self.t_mult = t_mult
        self.alpha_f = alpha_f

    def __call__(self, state: State, ssr: float = 1.0):
        t_0 = _convert_time(self.t_0, state, ssr=ssr)
        current_interval_len = t_0
        current_interval_end = t_0
        while current_interval_end <= state.timer.get(current_interval_end.unit):
            if current_interval_len.value == 0:
                raise ValueError(
                    'Interval between restarts for cosine annealing/warm restarts scheduler has decayed to 0.')

            current_interval_len = Time(value=int(self.t_mult * current_interval_len.value),
                                        unit=current_interval_len.unit)
            current_interval_end += current_interval_len

        current_interval_start = current_interval_end - current_interval_len
        frac_of_current_interval = ((state.timer.get(t_0.unit) - current_interval_start) / current_interval_len).value

        return _cosine_anneal(x=frac_of_current_interval, min_y=self.alpha_f)


class PolynomialScheduler(ComposerScheduler):
    r"""Sets the learning rate to be proportional to a power of the fraction of training time left.

    Specifially, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_f + (1 - \alpha_f) \times (1 - \tau) ^ {\kappa}

    Given :math:`\tau`, the fraction of time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau = t / t_{max}
    
    Where :math:`\kappa`
    represents the exponent to be used for the proportionality relationship, :math:`t_{max}` represents the duration of
    this scheduler, and :math:`\alpha_f` represents the learning rate multiplier to decay to.

    Args:
        power (float): The exponent to be used for the proportionality relationship.
        t_max (str or Time): The duration of this scheduler. Default = ``'1dur'``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
    """

    def __init__(self, power: float, t_max: Union[str, Time] = '1dur', alpha_f: float = 0.0):
        self.t_max = t_max
        self.power = power
        self.alpha_f = alpha_f

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(t_max.unit)
        frac_of_total = (current_time / t_max).value

        coeff = (1 - frac_of_total)**self.power
        current_factor = self.alpha_f + coeff * (1.0 - self.alpha_f)
        return current_factor


class MultiStepWithWarmupScheduler(ComposerScheduler):
    r"""Decays the learning rate discretely at fixed milestones, with a linear warmup.

    Starts with a linear warmup over ``t_warmup`` time, then decays the learning rate by a factor of ``gamma``
    whenever a time milestone in ``milestones`` is reached.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            \\ t / t_{warmup}, & \text{if } t < t_{warmup}
            \gamma ^ x & \text{otherwise}
        \end{cases}

    Where :math:`t_warmup` represents the warmup time, :math:`x` represents the amount of milestones that have been
    reached, and :math:`\gamma` represents the multiplicative decay factor.

    Args:
        t_warmup (str or Time): Warmup time.
        milestones (list of str or Time): Times at which the learning rate should change.
        gamma (float): Multiplicative decay factor. Default = ``0.1``.
    """

    def __init__(self, t_warmup: Union[str, Time], milestones: List[Union[str, Time]], gamma: float = 0.1):
        self.t_warmup = t_warmup
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)
        self.step_scheduler = MultiStepScheduler(milestones=milestones, gamma=gamma)

    def __call__(self, state: State, ssr: float = 1.0):
        # N.B. warmup time is intentionally *not* subject to scale schedule
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timer < t_warmup:
            return self.warmup_scheduler(state)

        return self.step_scheduler(state, ssr)


class LinearWithWarmupScheduler(ComposerScheduler):
    r"""Adjusts the learning rate linearly, with a linear warmup.

    Linearly adjusts the learning rate multiplier from ``alpha_i`` to ``alpha_f`` over ``t_max`` time.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}    
            \\ t / t_{warmup}, & \text{if } t < t_{warmup}
            \alpha_i + (alpha_f - \alpha_i) \times \tau_w & \text{otherwise}
        \end{cases}

    Given :math:`\tau_w`, the fraction of post-warmup time elpased (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau_w = (t - t_{warmup} / t_{max}
    
    Where :math:`t_warmup` represents the warmup time, :math:`\alpha_i` represents the initial learning rate multiplier, and :math:`\alpha_f`
    represents the learning rate multiplier to decay to, and :math:`t_max` represents the duration of this scheduler.

    Args:
        t_warmup (str or Time): Warmup time.
        alpha_i (float): Initial learning rate multiplier. Default = ``1.0``.
        alpha_f (float): Final learning rate multiplier. Default = ``0.0``.
        t_max (str or Time): The duration of this scheduler. Default = ``'1dur'``.
    """

    def __init__(self,
                 t_warmup: Union[str, Time],
                 alpha_i: float = 1.0,
                 alpha_f: float = 0.0,
                 t_max: Union[str, Time] = '1dur'):
        self.t_warmup = t_warmup
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_max = t_max
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=alpha_i, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        # N.B. warmup time is intentionally *not* subject to scale schedule
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timer < t_warmup:
            return self.warmup_scheduler(state)

        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(t_warmup.unit)
        frac_of_total = min(1.0, ((current_time - t_warmup) / (t_max - t_warmup)).value)

        current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)

        return current_factor


class CosineAnnealingWithWarmupScheduler(ComposerScheduler):
    r"""Decays the learning rate according to the decreasing part of a cosine curve, with a linear warmup.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}    
            \\ t / t_{warmup}, & \text{if } t < t_{warmup}
            \alpha_f + (1 - \alpha_f) \times \frac{1}{2} (1 + \cos(\pi \times \tau_w)) & \text{otherwise}
        \end{cases}

    Given :math:`\tau_w`, the fraction of post-warmup time elpased (clipped to the interval :math:`[0, 1]`), as:

    .. math::
       \tau_w = (t - t_{warmup} / t_{max}
    
    Where :math:`t_warmup` represents the warmup time, :math:`t_max` represents the duration of this scheduler, and :math:`\alpha_f` represents the learning rate multiplier to decay to.
    
    Args:
        t_warmup (str or Time): Warmup time.
        t_max (str or Time): The duration of this scheduler. Default = ``'1dur'``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
    """

    def __init__(self, t_warmup: Union[str, Time], t_max: Union[str, Time] = '1dur', alpha_f: float = 0.0):
        self.t_warmup = t_warmup
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        # N.B. warmup time is intentionally *not* subject to scale schedule
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timer < t_warmup:
            return self.warmup_scheduler(state)

        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(t_warmup.unit)
        frac_of_total = ((current_time - t_warmup) / (t_max - t_warmup)).value

        return _cosine_anneal(x=frac_of_total, min_y=self.alpha_f)
