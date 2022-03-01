# Copyright 2021 MosaicML. All Rights Reserved.

"""Hyperparameters for schedulers."""

import functools
from abc import ABC
from dataclasses import asdict, dataclass
from typing import List

import yahp as hp

from composer.optim.scheduler import (ComposerScheduler, constant_scheduler, cosine_annealing_scheduler,
                                      cosine_annealing_warm_restarts_scheduler, cosine_annealing_with_warmup_scheduler,
                                      exponential_scheduler, linear_scheduler, linear_with_warmup_scheduler,
                                      multi_step_scheduler, multi_step_with_warmup_scheduler, polynomial_scheduler,
                                      step_scheduler)


@dataclass
class SchedulerHparams(hp.Hparams, ABC):

    scheduler_function = None

    def initialize_object(self) -> ComposerScheduler:
        if self.scheduler_function is None:
            raise NotImplementedError(f"Cannot initialize {self} because `scheduler_function` is undefined.")

        return functools.partial(self.scheduler_function.__func__, **asdict(self))


@dataclass
class PolynomialLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`polynomial_scheduler` scheduler."""

    power: float = hp.required(doc='Power of LR schedule.')
    t_max: str = hp.optional(default='1dur', doc='Total scheduler duration.')
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate.')

    scheduler_function = polynomial_scheduler


@dataclass
class ConstantLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`constant_scheduler` scheduler."""

    factor: float = hp.optional(default=1.0, doc='Constant learning rate factor')
    total_time: str = hp.optional(default='1dur', doc='Total scheduler duration')

    scheduler_function = constant_scheduler


@dataclass
class StepLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`step_scheduler` scheduler."""

    step_size: str = hp.required(doc='Period of learning rate decay')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative factor of decay')

    scheduler_function = step_scheduler


@dataclass
class MultiStepLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`multi_step_scheduler` scheduler."""

    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative factor of decay')

    scheduler_function = multi_step_scheduler


@dataclass
class ExponentialLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`exponential_scheduler` scheduler."""

    gamma: float = hp.required(doc='Multiplicative factor of decay')

    scheduler_function = exponential_scheduler


@dataclass
class CosineAnnealingLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`cosine_annealing_scheduler` scheduler."""

    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')

    scheduler_function = cosine_annealing_scheduler


@dataclass
class CosineAnnealingWarmRestartsHparams(SchedulerHparams):
    """Hyperparameters for the :func:`cosine_annealing_warm_restarts_scheduler` scheduler."""

    t_0: str = hp.optional(default='1dur', doc="Duration for the first restart.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')
    t_mult: float = hp.optional(default=1.0, doc="A factor increases :math:`t_{i}` after a restart. Default: 1.")

    scheduler_function = cosine_annealing_warm_restarts_scheduler


@dataclass
class LinearLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`linear_scheduler` scheduler."""

    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=0.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_function = linear_scheduler


@dataclass
class MultiStepWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`multi_step_with_warmup_scheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')

    scheduler_function = multi_step_with_warmup_scheduler


@dataclass
class LinearWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`linear_with_warmup_scheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=0.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_function = linear_with_warmup_scheduler


@dataclass
class CosineAnnealingWithWarmupLRHparams(SchedulerHparams):
    """Hyperparameters for the :func:`cosine_annealing_with_warmup_scheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')

    scheduler_function = cosine_annealing_with_warmup_scheduler
