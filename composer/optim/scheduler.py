# Copyright 2021 MosaicML. All Rights Reserved.

import functools
import logging
import math
import warnings
from abc import ABC
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List, Union

import yahp as hp
from torch.optim.lr_scheduler import LambdaLR

from composer.core import State
from composer.core.time import Time, TimeUnit
from composer.core.types import Scheduler
from composer.optim._time_conversion import convert as convert_time

try:
    from typing import Protocol
except ImportError:
    Protocol = object  # Protocol is not available in python 3.7

if TYPE_CHECKING:
    from typing import Protocol

log = logging.getLogger(__name__)


class ComposerSchedulerFn(Protocol):
    """Specification for a "stateless" scheduler function.

    A scheduler function should be a pure function that returns a learning rate, given the current
    trainer state, and optionally a "scale schedule ratio" (SSR). A typical implementation will
    read `state.timer`, and possibly other fields like `state.max_duration`, to determine the
    trainer's latest temporal progress.
    """

    def __call__(self, state: State, *, ssr: float = 1.0) -> float:
        raise NotImplementedError


ComposerScheduler = Union[Scheduler, ComposerSchedulerFn]


def _convert_time(time: Union[str, Time], state: State, ssr: float = 1.0) -> Time[int]:
    if isinstance(time, str):
        time = Time.from_timestring(time)

    if time.unit == TimeUnit.DURATION:
        time = convert_time(time=time, unit=state.max_duration.unit, max_training_duration=state.max_duration)
    elif ssr != 1.0:
        time = Time(value=int(ssr * time.value), unit=time.unit)

    if time.unit == TimeUnit.EPOCH:
        time = convert_time(time=time, unit=TimeUnit.BATCH, steps_per_epoch=state.steps_per_epoch)

    return time


def compile_scheduler(scheduler: ComposerScheduler, state: State) -> Scheduler:

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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
    """

    total_time = _convert_time(total_time, state, ssr=ssr)

    if state.timer < total_time:
        return factor

    return 1.0


def linear_scheduler(state: State,
                     *,
                     ssr: float = 1.0,
                     start_factor: float = 1.0 / 3,
                     end_factor: float = 1.0,
                     total_time: Union[str, Time] = '1dur') -> float:
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
    """

    total_time = _convert_time(total_time, state, ssr=ssr)
    current_time = state.timer.get(total_time.unit)
    frac_of_total = min(1.0, (current_time / total_time).value)

    current_factor = start_factor + frac_of_total * (end_factor - start_factor)

    return current_factor


def exponential_scheduler(state: State, *, ssr: float = 1.0, gamma: float) -> float:
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
    """

    current_time = state.timer.epoch

    return gamma**(current_time.value / ssr)


def _cosine_anneal(x: float, min_y: float = 0, max_y: float = 1) -> float:
    """Implements a cosine decay curve.

    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range [min_y, max_y]. Additionally, param x is
    clipped to the interval [0, 1]
    """

    x = min(max(x, 0.0), 1.0)
    return min_y + (max_y - min_y) * (1 + math.cos(x * math.pi)) / 2


def cosine_annealing_scheduler(state: State,
                               *,
                               ssr: float = 1.0,
                               t_max: Union[str, Time] = '1dur',
                               min_factor: float = 0.0):
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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
    r"""Decays the learning rate of each parameter group by gamma every step_size time.
    
    Args:
        state (State): The current Composer Trainer state.
        ssr (float): The scale schedule ratio. In general, the learning rate computed by this
            scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
            this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.
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


@dataclass
class SchedulerHparams(hp.Hparams, ABC):

    scheduler_function = None

    def initialize_object(self) -> ComposerScheduler:
        if self.scheduler_function is None:
            raise NotImplementedError(f"Cannot initialize {self} because `scheduler_function` is undefined.")

        return functools.partial(self.scheduler_function.__func__, **asdict(self))


@dataclass
class PolynomialLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`PolynomialLR` scheduler."""
    power: float = hp.required(doc='Power of LR schedule.')
    t_max: str = hp.optional(default='1dur', doc='Maximum number of iterations.')
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate.')

    scheduler_function = polynomial_scheduler


@dataclass
class ConstantLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ConstantLR` scheduler."""

    factor: float = hp.optional(default=1.0, doc='foo')
    total_time: str = hp.optional(default='1dur', doc='foo')

    scheduler_function = constant_scheduler


@dataclass
class StepLRHparams(SchedulerHparams):
    """Hyperparameters for the `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#t
    orch.optim.lr_scheduler.StepLR>`_ scheduler."""

    step_size: str = hp.required(doc='Period of learning rate decay')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')

    scheduler_function = step_scheduler


@dataclass
class MultiStepLRHparams(SchedulerHparams):
    """Hyperparameters for the `MultiStepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiSte
    pLR.html#torch.optim.lr_scheduler.MultiStepLR>`_ scheduler."""

    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')

    scheduler_function = multi_step_scheduler


@dataclass
class ExponentialLRHparams(SchedulerHparams):
    """Hyperparameters for the `ExponentialLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.Expone
    ntialLR.html#torch.optim.lr_scheduler.ExponentialLR>`_ scheduler."""

    gamma: float = hp.required(doc='multiplicative factor of decay')

    scheduler_function = exponential_scheduler


@dataclass
class CosineAnnealingLRHparams(SchedulerHparams):
    """Hyperparameters for the `CosineAnnealingLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.Co
    sineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_ scheduler."""

    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='minimum learning rate factor.')

    scheduler_function = cosine_annealing_scheduler


@dataclass
class CosineAnnealingWarmRestartsHparams(SchedulerHparams):
    """Hyperparameters for the `CosineAnnealingWarmRestarts <https://pytorch.org/docs/stable/generated/torch.optim.lr_
    scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_ scheduler."""

    t_0: str = hp.optional(default='1dur', doc="Duration for the first restart.")
    min_factor: float = hp.optional(default=0.0, doc='minimum learning rate.')
    t_mult: float = hp.optional(default=1.0, doc="A factor increases :math:`t_{i}` after a restart. Default: 1.")

    scheduler_function = cosine_annealing_warm_restarts_scheduler


@dataclass
class LinearLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~torch.optim.lr_scheduler.LinearLR` scheduler."""

    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0 / 3)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=1.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_function = linear_scheduler


@dataclass
class MultiStepWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the ``CosineAnnealingWarmRestarts` <https://pytorch.org/docs/stable/generated/torch.optim.lr_
    scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_ scheduler."""

    warmup_time: str = hp.required(doc='foo')
    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')

    scheduler_function = multi_step_with_warmup_scheduler


@dataclass
class LinearWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the ``CosineAnnealingWarmRestarts` <https://pytorch.org/docs/stable/generated/torch.optim.lr_
    scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_ scheduler."""

    warmup_time: str = hp.required(doc='foo')
    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0 / 3)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=1.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_function = linear_with_warmup_scheduler


@dataclass
class CosineAnnealingWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the ``CosineAnnealingWarmRestarts` <https://pytorch.org/docs/stable/generated/torch.optim.lr_
    scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_ scheduler."""

    warmup_time: str = hp.required(doc='foo')
    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='minimum learning rate factor.')

    scheduler_function = cosine_annealing_with_warmup_scheduler
