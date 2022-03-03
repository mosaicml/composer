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
    """Abstract base class for scheduler hyperparameter classes."""

    scheduler_cls = None  # type: Optional[Type[ComposerScheduler]]

    def initialize_object(self) -> ComposerScheduler:
        if self.scheduler_cls is None:
            raise NotImplementedError(f"Cannot initialize {self} because `scheduler_cls` is undefined.")

        # Expected no arguments to "ComposerScheduler" constructor
        return self.scheduler_cls(**asdict(self))  # type: ignore


@dataclass
class StepSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`StepScheduler` scheduler."""

    step_size: str = hp.required(doc='Time between changes to the learning rate.')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative decay factor.')

    scheduler_cls = StepScheduler


@dataclass
class MultiStepSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`MultiStepScheduler` scheduler."""

    milestones: List[str] = hp.required(doc='Times at which the learning rate should change.')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative decay factor.')

    scheduler_cls = MultiStepScheduler


@dataclass
class ConstantSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ConstantScheduler` scheduler."""

    alpha: float = hp.optional(default=1.0, doc='Learning rate multiplier to maintain while this scheduler is active.')
    t_max: str = hp.optional(default='1dur', doc='Duration of this scheduler.')

    scheduler_cls = ConstantScheduler


@dataclass
class LinearSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`LinearScheduler` scheduler."""

    alpha_i: float = hp.optional("Initial learning rate multiplier.", default=1.0)
    alpha_f: float = hp.optional("Final learning rate multiplier.", default=0.0)
    t_max: str = hp.optional(default='1dur', doc='Duration of this scheduler.')

    scheduler_cls = LinearScheduler


@dataclass
class ExponentialSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ExponentialScheduler` scheduler."""

    gamma: float = hp.required(doc='Multiplicative decay factor.')
    decay_period: str = hp.optional(default='1ep', doc='Decay period.')

    scheduler_cls = ExponentialScheduler


@dataclass
class CosineAnnealingSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingScheduler` scheduler."""

    t_max: str = hp.optional(default='1dur', doc='Duration of this scheduler.')
    alpha_f: float = hp.optional(default=0.0, doc='Learning rate multiplier to decay to.')

    scheduler_cls = CosineAnnealingScheduler


@dataclass
class CosineAnnealingWarmRestartsSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingWarmRestartsScheduler` scheduler."""

    t_0: str = hp.optional(default='1dur', doc="The period of the first cycle.")
    alpha_f: float = hp.optional(default=0.0, doc='Learning rate multiplier to decay to.')
    t_mult: float = hp.optional(default=1.0, doc="The multiplier for the duration of successive cycles.")

    scheduler_cls = CosineAnnealingWarmRestartsScheduler


@dataclass
class PolynomialSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`PolynomialScheduler` scheduler."""

    power: float = hp.required(doc='The exponent to be used for the proportionality relationship.')
    t_max: str = hp.optional(default='1dur', doc='Duration of this scheduler.')
    alpha_f: float = hp.optional(default=0.0, doc='Learning rate multiplier to decay to.')

    scheduler_cls = PolynomialScheduler


@dataclass
class MultiStepWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`MultiStepWithWarmupScheduler` scheduler."""

    t_warmup: str = hp.required(doc='Warmup time.')
    milestones: List[str] = hp.required(doc='Times at which the learning rate should change.')
    gamma: float = hp.optional(default=0.1, doc='Multiplicative decay factor.')

    scheduler_cls = MultiStepWithWarmupScheduler


@dataclass
class LinearWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`LinearWithWarmupScheduler` scheduler."""

    t_warmup: str = hp.required(doc='Warmup time.')
    alpha_i: float = hp.optional("Initial learning rate multiplier.", default=1.0)
    alpha_f: float = hp.optional("Final learning rate multiplier.", default=0.0)
    t_max: str = hp.optional(default='1dur', doc='Duration of this scheduler.')

    scheduler_cls = LinearWithWarmupScheduler


@dataclass
class CosineAnnealingWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`CosineAnnealingWithWarmupScheduler` scheduler."""

    t_warmup: str = hp.required(doc='Warmup time.')
    t_max: str = hp.optional(default='1dur', doc='Duration of this scheduler.')
    alpha_f: float = hp.optional(default=0.0, doc='Learning rate multiplier to decay to.')

    scheduler_cls = CosineAnnealingWithWarmupScheduler
