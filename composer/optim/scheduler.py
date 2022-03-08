# Copyright 2021 MosaicML. All Rights Reserved.

import inspect
import logging
import math
import textwrap
import warnings
from abc import ABC
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List, Optional, Type, Union

import yahp as hp
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
    
    Args:
        step_size (str, Time): TODO.
        gamma (float): Gamma. Default = ``0.1``.
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
    
    Args:
        milestones (list of str or Time): Milestones.
        gamma (float) Gamma. Default = ``0.1``.
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
    
    Args:
        factor (float): Factor. Default = ``1.0``.
        total_time (str or Time): Total time. Default = ``'1dur'``.
    """

    def __init__(self, factor: float = 1.0, total_time: Union[str, Time] = '1dur') -> None:
        self.factor = factor
        self.total_time = total_time

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        total_time = _convert_time(self.total_time, state, ssr=ssr)

        if state.timer < total_time:
            return self.factor

        return 1.0


class LinearScheduler(ComposerScheduler):
    r"""Adjusts the learning rate linearly.
    
    Args:
        start_factor (float): Start factor. Default = ``1.0``.
        end_factor (float): End factor. Default = ``0.0``.
        total_time (str or Time): Total time. Default = ``'1dur'``.
    """

    def __init__(self, start_factor: float = 1.0, end_factor: float = 0.0, total_time: Union[str, Time] = '1dur'):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_time = Time.from_timestring(total_time) if isinstance(total_time, str) else total_time

    def __call__(self, state: State, ssr: float = 1.0):
        total_time = _convert_time(self.total_time, state, ssr=ssr)
        current_time = state.timer.get(total_time.unit)
        frac_of_total = min(1.0, (current_time / total_time).value)

        current_factor = self.start_factor + frac_of_total * (self.end_factor - self.start_factor)

        return current_factor


class ExponentialScheduler(ComposerScheduler):
    """Decays the learning rate exponentially.

    Args:
        gamma (float): Gamma.
        decay_period (str or Time, optional): The period over which to decay the learning rate by gamma.
            The learning rate is approximately calculated as ``lr = gamma ** (current_time / decay_period)``.
            (default: ``'1ep'``)
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
    """Decays the learning rate according to the decreasing part of a cosine curve.

    Args:
        t_max (str or Time): Total time. Default = ``'1dur'``.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    def __init__(self, t_max: Union[str, Time] = '1dur', min_factor: float = 0.0):
        self.t_max = t_max
        self.min_factor = min_factor

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(t_max.unit)
        frac_of_total = (current_time / t_max).value

        return _cosine_anneal(x=frac_of_total, min_y=self.min_factor)


class CosineAnnealingWarmRestartsScheduler(ComposerScheduler):
    """Cyclically decays the learning rate according to the decreasing part of a cosine curve.

    Args:
        t_0 (str or Time): The first cycle's duration.
        t_mult (float): The multiplier for subsequent cycles' durations. Default = ``1.0``.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    def __init__(self, t_0: Union[str, Time], t_mult: float = 1.0, min_factor: float = 0.0):
        self.t_0 = t_0
        self.t_mult = t_mult
        self.min_factor = min_factor

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

        return _cosine_anneal(x=frac_of_current_interval, min_y=self.min_factor)


class PolynomialScheduler(ComposerScheduler):
    """Sets the learning rate to be exponentially proportional to the percentage of training time left.

    Args:
        power (float): Power.
        t_max (str or Time): Total time. Default = ``'1dur'``.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    def __init__(self, power: float, t_max: Union[str, Time] = '1dur', min_factor: float = 0.0):
        self.t_max = t_max
        self.power = power
        self.min_factor = min_factor

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(t_max.unit)
        frac_of_total = (current_time / t_max).value

        coeff = (1 - frac_of_total)**self.power
        current_factor = self.min_factor + coeff * (1.0 - self.min_factor)
        return current_factor


class MultiStepWithWarmupScheduler(ComposerScheduler):
    """Decays the learning rate discretely at fixed milestones, with a linear warmup.

    Args:
        warmup_time (str or Time): Warmup time.
        milestones (list of str or Time): Milestones.
        gamma (float) Gamma. Default = ``0.1``.
    """

    def __init__(self, warmup_time: Union[str, Time], milestones: List[Union[str, Time]], gamma: float = 0.1):
        self.warmup_time = warmup_time
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_scheduler = LinearScheduler(start_factor=0.0, end_factor=1.0, total_time=warmup_time)
        self.step_scheduler = MultiStepScheduler(milestones=milestones, gamma=gamma)

    def __call__(self, state: State, ssr: float = 1.0):
        # N.B. warmup time is intentionally *not* subject to scale schedule
        warmup_time = _convert_time(self.warmup_time, state)
        if warmup_time.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timer < warmup_time:
            return self.warmup_scheduler(state)

        return self.step_scheduler(state, ssr)


class LinearWithWarmupScheduler(ComposerScheduler):
    """Adjusts the learning rate linearly, with a linear warmup.

    Args:
        warmup_time (str or Time): Warmup time.
        start_factor (float): Start factor. Default = ``1.0``.
        end_factor (float): End factor. Default = ``0.0``.
        total_time (str or Time): Total time. Default = ``'1dur'``.
    """

    def __init__(self,
                 warmup_time: Union[str, Time],
                 start_factor: float = 1.0,
                 end_factor: float = 0.0,
                 total_time: Union[str, Time] = '1dur'):
        self.warmup_time = warmup_time
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_time = total_time
        self.warmup_scheduler = LinearScheduler(start_factor=0.0, end_factor=start_factor, total_time=warmup_time)

    def __call__(self, state: State, ssr: float = 1.0):
        # N.B. warmup time is intentionally *not* subject to scale schedule
        warmup_time = _convert_time(self.warmup_time, state)
        if warmup_time.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timer < warmup_time:
            return self.warmup_scheduler(state)

        total_time = _convert_time(self.total_time, state, ssr=ssr)
        current_time = state.timer.get(warmup_time.unit)
        frac_of_total = min(1.0, ((current_time - warmup_time) / (total_time - warmup_time)).value)

        current_factor = self.start_factor + frac_of_total * (self.end_factor - self.start_factor)

        return current_factor


class CosineAnnealingWithWarmupScheduler(ComposerScheduler):
    r"""Decays the learning rate according to the decreasing part of a cosine curve, with a linear warmup.
    
    Args:
        warmup_time (str or Time): Warmup time.
        t_max (str or Time): Total time. Default = ``'1dur'``.
        min_factor (float): Minimum factor. Default = ``0.0``.
    """

    def __init__(self, warmup_time: Union[str, Time], t_max: Union[str, Time] = '1dur', min_factor: float = 0.0):
        self.warmup_time = warmup_time
        self.t_max = t_max
        self.min_factor = min_factor
        self.warmup_scheduler = LinearScheduler(start_factor=0.0, end_factor=1.0, total_time=warmup_time)

    def __call__(self, state: State, ssr: float = 1.0):
        # N.B. warmup time is intentionally *not* subject to scale schedule
        warmup_time = _convert_time(self.warmup_time, state)
        if warmup_time.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timer < warmup_time:
            return self.warmup_scheduler(state)

        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timer.get(warmup_time.unit)
        frac_of_total = ((current_time - warmup_time) / (t_max - warmup_time)).value

        return _cosine_anneal(x=frac_of_total, min_y=self.min_factor)


@dataclass
class SchedulerHparams(hp.Hparams, ABC):

    scheduler_cls = None  # type: Optional[Type[ComposerScheduler]]

    def initialize_object(self) -> ComposerScheduler:
        if self.scheduler_cls is None:
            raise NotImplementedError(f"Cannot initialize {self} because `scheduler_cls` is undefined.")

        # Expected no arguments to "ComposerScheduler" constructor
        return self.scheduler_cls(**asdict(self))  # type: ignore


@dataclass
class PolynomialLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`PolynomialScheduler` scheduler."""

    power: float = hp.required(doc='Power of LR schedule.')
    t_max: str = hp.optional(default='1dur', doc='Total scheduler duration.')
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate.')

    scheduler_cls = PolynomialScheduler


@dataclass
class ConstantLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ConstantScheduler` scheduler."""

    factor: float = hp.optional(default=1.0, doc='Constant learning rate factor')
    total_time: str = hp.optional(default='1dur', doc='Total scheduler duration')

    scheduler_cls = ConstantScheduler


@dataclass
class StepLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`StepScheduler` scheduler."""

    step_size: str = hp.required(doc='Period of learning rate decay')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative factor of decay')

    scheduler_cls = StepScheduler


@dataclass
class MultiStepLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`MultiStepScheduler` scheduler."""

    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative factor of decay')

    scheduler_cls = MultiStepScheduler


@dataclass
class ExponentialLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ExponentialScheduler` scheduler."""

    gamma: float = hp.required(doc='Multiplicative factor of decay')
    decay_period: str = hp.optional(default='1ep', doc='Decay period')

    scheduler_cls = ExponentialScheduler


@dataclass
class CosineAnnealingLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingScheduler` scheduler."""

    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')

    scheduler_cls = CosineAnnealingScheduler


@dataclass
class CosineAnnealingWarmRestartsHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingWarmRestartsScheduler` scheduler."""

    t_0: str = hp.optional(default='1dur', doc="Duration for the first restart.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')
    t_mult: float = hp.optional(default=1.0, doc="A factor increases :math:`t_{i}` after a restart. Default: 1.")

    scheduler_cls = CosineAnnealingWarmRestartsScheduler


@dataclass
class LinearLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`LinearScheduler` scheduler."""

    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=0.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_cls = LinearScheduler


@dataclass
class MultiStepWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`MultiStepWithWarmupScheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')

    scheduler_cls = MultiStepWithWarmupScheduler


@dataclass
class LinearWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`LinearWithWarmupScheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=0.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_cls = LinearWithWarmupScheduler


@dataclass
class CosineAnnealingWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingWithWarmupScheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')

    scheduler_cls = CosineAnnealingWithWarmupScheduler
