# Copyright 2021 MosaicML. All Rights Reserved.

import functools
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Optional, Union

import yahp as hp

from composer.core import State
from composer.core.scheduler import ComposerSchedulerFn
from composer.core.time import Time, TimeUnit
from composer.utils._time_conversion import convert as convert_time

log = logging.getLogger(__name__)


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


def step_scheduler(state: State, *, ssr: float = 1.0, step_size: Union[str, Time], gamma: float = 0.1) -> float:
    step_size = _convert_time(step_size, state, ssr=ssr)
    print(step_size)
    current_time = state.timer.get(step_size.unit)
    steps = int(current_time / step_size)

    return gamma**steps


def multi_step_scheduler(state: State,
                         *,
                         ssr: float = 1.0,
                         milestones: List[Union[str, Time]],
                         gamma: float = 0.1) -> float:
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
    total_time = _convert_time(total_time, state, ssr=ssr)
    current_time = state.timer.get(total_time.unit)
    frac_of_total = min(1.0, (current_time / total_time).value)

    current_factor = start_factor + frac_of_total * (end_factor - start_factor)

    return current_factor


def exponential_scheduler(state: State, *, ssr: float = 1.0, gamma: float) -> float:
    current_time = state.timer.epoch

    return gamma**(current_time.value / ssr)


def _cosine_anneal(x: float, min_y: float = 0, max_y: float = 1) -> float:
    """Implements a cosine decay curve.
    
    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range [min_y, max_y].
    Additionally, param x is clipped to the interval [0, 1]
    """

    x = min(max(x, 0.0), 1.0)
    return min_y + (max_y - min_y) * (1 + math.cos(x * math.pi)) / 2


def cosine_annealing_scheduler(state: State,
                               *,
                               ssr: float = 1.0,
                               T_max: Union[str, Time] = '1dur',
                               min_factor: float = 0.0):
    T_max = _convert_time(T_max, state, ssr=ssr)
    current_time = state.timer.get(T_max.unit)
    frac_of_total = (current_time / T_max).value

    return _cosine_anneal(x=frac_of_total, min_y=min_factor)


def cosine_annealing_warm_restarts_scheduler(state: State,
                                             *,
                                             ssr: float = 1.0,
                                             T_0: Union[str, Time],
                                             T_mult: float = 1.0,
                                             min_factor: float = 0.0):
    T_0 = _convert_time(T_0, state, ssr=ssr)
    current_interval_len = T_0
    current_interval_end = T_0
    while current_interval_end <= state.timer.get(current_interval_end.unit):
        if current_interval_len.value == 0:
            raise ValueError('Interval between restarts for cosine annealing/warm restarts scheduler has decayed to 0.')

        current_interval_len = Time(value=int(T_mult * current_interval_len.value), unit=current_interval_len.unit)
        current_interval_end += current_interval_len

    current_interval_start = current_interval_end - current_interval_len
    frac_of_current_interval = ((state.timer.get(T_0.unit) - current_interval_start) / current_interval_len).value

    return _cosine_anneal(x=frac_of_current_interval, min_y=min_factor)


def polynomial_scheduler(state: State,
                         *,
                         ssr: float = 1.0,
                         T_max: Union[str, Time] = '1dur',
                         power: float,
                         min_factor: float = 0.0):
    T_max = _convert_time(T_max, state, ssr=ssr)
    current_time = state.timer.get(T_max.unit)
    frac_of_total = (current_time / T_max).value

    coeff = (1 - frac_of_total)**power
    current_factor = min_factor + coeff * (1.0 - min_factor)
    return current_factor


def multi_step_with_warmup_scheduler(state: State,
                                     *,
                                     ssr: float = 1.0,
                                     warmup_time: Union[str, Time],
                                     milestones: List[Union[str, Time]],
                                     gamma: float = 0.1) -> float:
    warmup_time = _convert_time(warmup_time, state)
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
    # N.B. warmup time is intentionally *not* subject to scale schedule
    warmup_time = _convert_time(warmup_time, state)
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
                                           T_max: Union[str, Time] = '1dur',
                                           min_factor: float = 0.0):
    # N.B. warmup time is intentionally *not* subject to scale schedule
    warmup_time = _convert_time(warmup_time, state)
    if state.timer < warmup_time:
        return linear_scheduler(state, start_factor=0.0, end_factor=1.0, total_time=warmup_time)

    T_max = _convert_time(T_max, state, ssr=ssr)
    current_time = state.timer.get(warmup_time.unit)
    frac_of_total = ((current_time - warmup_time) / (T_max - warmup_time)).value

    return _cosine_anneal(x=frac_of_total, min_y=min_factor)


def get_scheduler(scheduler: ComposerSchedulerFn, ssr: float = 1.0, **kwargs) -> ComposerSchedulerFn:
    return functools.partial(scheduler, ssr=ssr, **kwargs)


@dataclass
class SchedulerHparams(hp.Hparams, ABC):

    scheduler_function = None

    def initialize_object(self) -> ComposerSchedulerFn:
        if self.scheduler_function is None:
            raise NotImplementedError(f"Cannot initialize {self} because `scheduler_function` is undefined.")

        return functools.partial(self.scheduler_function.__func__, **asdict(self))


@dataclass
class PolynomialLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`PolynomialLR` scheduler."""
    power: float = hp.required(doc='Power of LR schedule.')
    T_max: str = hp.optional(default='1dur', doc='Maximum number of iterations.')
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
    """Hyperparameters for the `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR>`_
    scheduler.
    """

    step_size: str = hp.required(doc='Period of learning rate decay')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')

    scheduler_function = step_scheduler


@dataclass
class MultiStepLRHparams(SchedulerHparams):
    """Hyperparameters for the `MultiStepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR>`_ 
    scheduler.
    """

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

    T_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='minimum learning rate factor.')

    scheduler_function = cosine_annealing_scheduler


@dataclass
class CosineAnnealingWarmRestartsHparams(SchedulerHparams):
    """Hyperparameters for the ``CosineAnnealingWarmRestarts` <https://pytorch.org/docs/stable/generated/torch.optim.lr_
    scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_ scheduler."""

    T_0: str = hp.optional(default='1dur', doc="Duration for the first restart.")
    min_factor: float = hp.optional(default=0.0, doc='minimum learning rate.')
    T_mult: float = hp.optional(default=1.0, doc="A factor increases :math:`T_{i}` after a restart. Default: 1.")

    scheduler_function = cosine_annealing_warm_restarts_scheduler


@dataclass
class LinearLRHparams(SchedulerHparams):
    """Hyperparameters for the `LinearLRHparams.

    <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html>`_ scheduler.
    """

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
    T_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='minimum learning rate factor.')

    scheduler_function = cosine_annealing_with_warmup_scheduler
