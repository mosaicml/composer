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

__all__ = [
    "SchedulerHparams", "StepSchedulerHparams", "MultiStepSchedulerHparams", "ConstantSchedulerHparams",
    "LinearSchedulerHparams", "ExponentialSchedulerHparams", "CosineAnnealingSchedulerHparams",
    "CosineAnnealingWarmRestartsSchedulerHparams", "PolynomialSchedulerHparams", "MultiStepWithWarmupSchedulerHparams",
    "LinearWithWarmupSchedulerHparams", "CosineAnnealingWithWarmupSchedulerHparams"
]


@dataclass
class SchedulerHparams(hp.Hparams, ABC):
    """Base class for scheduler hyperparameter classes.

    Scheduler parameters that are added to :class:`~composer.trainer.trainer_hparams.TrainerHparams` (e.g. via YAML or
    the CLI) are initialized in the training loop.
    """

    _scheduler_cls = None  # type: Optional[Type[ComposerScheduler]]

    def initialize_object(self) -> ComposerScheduler:
        """Initializes the scheduler."""

        if self._scheduler_cls is None:
            raise NotImplementedError(f"Cannot initialize {self} because `_scheduler_cls` is undefined.")

        # Expected no arguments to "ComposerScheduler" constructor
        return self._scheduler_cls(**asdict(self))  # type: ignore


@dataclass
class StepSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.StepScheduler` scheduler.

    See :class:`~.StepScheduler` for documentation.

    Args:
        step_size (str, optional): See :class:`~.StepScheduler`.
        gamma (float, optional): See :class:`~.StepScheduler`.
    """

    step_size: str = hp.required(doc="Time between changes to the learning rate.")
    gamma: float = hp.optional(default=0.1, doc="Multiplicative decay factor.")

    _scheduler_cls = StepScheduler


@dataclass
class MultiStepSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.MultiStepScheduler` scheduler.

    See :class:`~.MultiStepScheduler` for documentation.

    Args:
        milestones (List[str]): See :class:`~.MultiStepScheduler`.
        gamma (float, optional): See :class:`~.MultiStepScheduler`.
    """

    milestones: List[str] = hp.required(doc="Times at which the learning rate should change.")
    gamma: float = hp.optional(default=0.1, doc="Multiplicative decay factor.")

    _scheduler_cls = MultiStepScheduler


@dataclass
class ConstantSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.ConstantScheduler` scheduler.

    See :class:`~.ConstantScheduler` for documentation.

    Args:
        alpha (float, optional): See :class:`~.ConstantScheduler`.
        t_max (str, optional): See :class:`~.ConstantScheduler`.
    """

    alpha: float = hp.optional(default=1.0, doc="Learning rate multiplier to maintain while this scheduler is active.")
    t_max: str = hp.optional(default="1dur", doc="Duration of this scheduler.")

    _scheduler_cls = ConstantScheduler


@dataclass
class LinearSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.LinearScheduler` scheduler.

    See :class:`~.LinearScheduler` for documentation.

    Args:
        alpha_i (float, optional): See :class:`~.LinearScheduler`.
        alpha_f (float, optional): See :class:`~.LinearScheduler`.
        t_max (str, optional): See :class:`~.LinearScheduler`.
    """

    alpha_i: float = hp.optional("Initial learning rate multiplier.", default=1.0)
    alpha_f: float = hp.optional("Final learning rate multiplier.", default=0.0)
    t_max: str = hp.optional(default="1dur", doc="Duration of this scheduler.")

    _scheduler_cls = LinearScheduler


@dataclass
class ExponentialSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.ExponentialScheduler` scheduler.

    See :class:`~.ExponentialScheduler` for documentation.

    Args:
        gamma (float): See :class:`~.ExponentialScheduler`.
        decay_period (str, optional): See :class:`~.ExponentialScheduler`.
    """

    gamma: float = hp.required(doc="Multiplicative decay factor.")
    decay_period: str = hp.optional(default="1ep", doc="Decay period.")

    _scheduler_cls = ExponentialScheduler


@dataclass
class CosineAnnealingSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.CosineAnnealingScheduler` scheduler.

    See :class:`~.CosineAnnealingScheduler` for documentation.

    Args:
        t_max (str, optional): See :class:`~.CosineAnnealingScheduler`.
        alpha_f (float, optional): See :class:`~.CosineAnnealingScheduler`.
    """

    t_max: str = hp.optional(default="1dur", doc="Duration of this scheduler.")
    alpha_f: float = hp.optional(default=0.0, doc="Learning rate multiplier to decay to.")

    _scheduler_cls = CosineAnnealingScheduler


@dataclass
class CosineAnnealingWarmRestartsSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.CosineAnnealingWarmRestartsScheduler` scheduler.

    See :class:`~.CosineAnnealingWarmRestartsScheduler` for documentation.

    Args:
        t_0 (str, optional): See :class:`~.CosineAnnealingWarmRestartsScheduler`.
        alpha_f (float, optional): See :class:`~.CosineAnnealingWarmRestartsScheduler`.
        t_mult (float, optional): See :class:`~.CosineAnnealingWarmRestartsScheduler`.
    """

    t_0: str = hp.optional(default="1dur", doc="The period of the first cycle.")
    alpha_f: float = hp.optional(default=0.0, doc="Learning rate multiplier to decay to.")
    t_mult: float = hp.optional(default=1.0, doc="The multiplier for the duration of successive cycles.")

    _scheduler_cls = CosineAnnealingWarmRestartsScheduler


@dataclass
class PolynomialSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.PolynomialScheduler` scheduler.

    See :class:`~.PolynomialScheduler` for documentation.

    Args:
        power (float): See :class:`~.PolynomialScheduler`.
        t_max (str, optional): See :class:`~.PolynomialScheduler`.
        alpha_f (float, optional): See :class:`~.PolynomialScheduler`.
    """

    power: float = hp.required(doc="The exponent to be used for the proportionality relationship.")
    t_max: str = hp.optional(default="1dur", doc="Duration of this scheduler.")
    alpha_f: float = hp.optional(default=0.0, doc="Learning rate multiplier to decay to.")

    _scheduler_cls = PolynomialScheduler


@dataclass
class MultiStepWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.MultiStepWithWarmupScheduler` scheduler.

    See :class:`~.MultiStepWithWarmupScheduler` for documentation.

    Args:
        t_warmup (str,): See :class:`~.MultiStepWithWarmupScheduler`.
        milestones (List[str]): See :class:`~.MultiStepWithWarmupScheduler`.
        gamma (float, optional): See :class:`~.MultiStepWithWarmupScheduler`.
    """

    t_warmup: str = hp.required(doc="Warmup time.")
    milestones: List[str] = hp.required(doc="Times at which the learning rate should change.")
    gamma: float = hp.optional(default=0.1, doc="Multiplicative decay factor.")

    _scheduler_cls = MultiStepWithWarmupScheduler


@dataclass
class LinearWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.LinearWithWarmupScheduler` scheduler.

    See :class:`~.LinearWithWarmupScheduler` for documentation.

    Args:
        t_warmup (str): See :class:`~.LinearWithWarmupScheduler`.
        alpha_i (float, optional): See :class:`~.LinearWithWarmupScheduler`.
        alpha_f (float, optional): See :class:`~.LinearWithWarmupScheduler`.
        t_max (str, optional): See :class:`~.LinearWithWarmupScheduler`.
    """

    t_warmup: str = hp.required(doc="Warmup time.")
    alpha_i: float = hp.optional("Initial learning rate multiplier.", default=1.0)
    alpha_f: float = hp.optional("Final learning rate multiplier.", default=0.0)
    t_max: str = hp.optional(default="1dur", doc="Duration of this scheduler.")

    _scheduler_cls = LinearWithWarmupScheduler


@dataclass
class CosineAnnealingWithWarmupSchedulerHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~.CosineAnnealingWithWarmupScheduler` scheduler.

    See :class:`~.CosineAnnealingWithWarmupScheduler` for documentation.

    Args:
        t_warmup (str): See :class:`~.CosineAnnealingWithWarmupScheduler`.
        t_max (str, optional): See :class:`~.CosineAnnealingWithWarmupScheduler`.
        alpha_f (float, optional): See :class:`~.CosineAnnealingWithWarmupScheduler`.
    """

    t_warmup: str = hp.required(doc="Warmup time.")
    t_max: str = hp.optional(default="1dur", doc="Duration of this scheduler.")
    alpha_f: float = hp.optional(default=0.0, doc="Learning rate multiplier to decay to.")

    _scheduler_cls = CosineAnnealingWithWarmupScheduler
