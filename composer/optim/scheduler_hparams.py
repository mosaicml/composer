# Copyright 2021 MosaicML. All Rights Reserved.

"""Hyperparameters for schedulers."""

from abc import ABC
from dataclasses import asdict, dataclass
from typing import List, Optional, Type

import yahp as hp

from composer.optim.scheduler import (ComposerScheduler, ConstantScheduler, CosineAnnealingScheduler,
                                      CosineAnnealingWarmRestartsScheduler, CosineAnnealingWithWarmupScheduler,
                                      ExponentialScheduler, LinearScheduler, LinearWithWarmupScheduler,
                                      MultiStepScheduler, MultiStepWithWarmupScheduler, PolynomialScheduler,
                                      StepScheduler)


@dataclass
class SchedulerHparams(hp.Hparams, ABC):

    scheduler_cls = None  # type: Optional[Type[ComposerScheduler]]

    def initialize_object(self) -> ComposerScheduler:
        if self.scheduler_cls is None:
            raise NotImplementedError(f"Cannot initialize {self} because `scheduler_cls` is undefined.")

        # Expected no arguments to "ComposerScheduler" constructor
        return self.scheduler_cls(**asdict(self))  # type: ignore


@dataclass
class StepSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`StepScheduler` scheduler."""

    step_size: str = hp.required(doc='Period of learning rate decay')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative factor of decay')

    scheduler_cls = StepScheduler


@dataclass
class MultiStepSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`MultiStepScheduler` scheduler."""

    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative factor of decay')

    scheduler_cls = MultiStepScheduler


@dataclass
class ConstantSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ConstantScheduler` scheduler."""

    factor: float = hp.optional(default=1.0, doc='Constant learning rate factor')
    total_time: str = hp.optional(default='1dur', doc='Total scheduler duration')

    scheduler_cls = ConstantScheduler


@dataclass
class LinearSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`LinearScheduler` scheduler."""

    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=0.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_cls = LinearScheduler


@dataclass
class ExponentialSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ExponentialScheduler` scheduler."""

    gamma: float = hp.required(doc='Multiplicative factor of decay')
    decay_period: str = hp.optional(default='1ep', doc='Decay period')

    scheduler_cls = ExponentialScheduler


@dataclass
class CosineAnnealingSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingScheduler` scheduler."""

    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')

    scheduler_cls = CosineAnnealingScheduler


@dataclass
class CosineAnnealingWarmRestartsSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingWarmRestartsScheduler` scheduler."""

    t_0: str = hp.optional(default='1dur', doc="Duration for the first restart.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')
    t_mult: float = hp.optional(default=1.0, doc="A factor increases :math:`t_{i}` after a restart. Default: 1.")

    scheduler_cls = CosineAnnealingWarmRestartsScheduler


@dataclass
class PolynomialSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`PolynomialScheduler` scheduler."""

    power: float = hp.required(doc='Power of LR schedule.')
    t_max: str = hp.optional(default='1dur', doc='Total scheduler duration.')
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate.')

    scheduler_cls = PolynomialScheduler


@dataclass
class MultiStepWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`MultiStepWithWarmupScheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')

    scheduler_cls = MultiStepWithWarmupScheduler


@dataclass
class LinearWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`LinearWithWarmupScheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0)
    end_factor: float = hp.optional("Number to multiply learning rate at the end.", default=0.0)
    total_time: str = hp.optional("Duration of linear decay steps. Default: full training duration.", default="1dur")

    scheduler_cls = LinearWithWarmupScheduler


@dataclass
class CosineAnnealingWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingWithWarmupScheduler` scheduler."""

    warmup_time: str = hp.required(doc='Warmup time')
    t_max: str = hp.optional(default='1dur', doc="Maximum scheduler duration.")
    min_factor: float = hp.optional(default=0.0, doc='Minimum learning rate factor.')

    scheduler_cls = CosineAnnealingWithWarmupScheduler
