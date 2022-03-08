# Copyright 2021 MosaicML. All Rights Reserved.

"""Optimizers and learning rate schedulers.

Composer is compatible with optimizers based off of PyTorch's native :class:`~torch.optim.Optimizer` API, and common
optimizers such as :class:`~torch.optim.SGD` and :class:`~torch.optim.Adam` have been thoroughly tested with Composer.
However, where applicable, it is recommended to use the optimizers provided in :mod:`~.decoupled_weight_decay` since
they improve off of their PyTorch equivalents.

PyTorch schedulers can be used with Composer, but this is explicitly discouraged. Instead, it is recommended to use
schedulers based off of Composer's :class:`~.scheduler.ComposerScheduler` API, which allows more flexibility and
configuration in writing schedulers.
"""

from composer.optim.decoupled_weight_decay import DecoupledAdamW as DecoupledAdamW
from composer.optim.decoupled_weight_decay import DecoupledSGDW as DecoupledSGDW
from composer.optim.optimizer_hparams import AdamHparams as AdamHparams
from composer.optim.optimizer_hparams import AdamWHparams as AdamWHparams
from composer.optim.optimizer_hparams import DecoupledAdamWHparams as DecoupledAdamWHparams
from composer.optim.optimizer_hparams import DecoupledSGDWHparams as DecoupledSGDWHparams
from composer.optim.optimizer_hparams import OptimizerHparams as OptimizerHparams
from composer.optim.optimizer_hparams import RAdamHparams as RAdamHparams
from composer.optim.optimizer_hparams import RMSpropHparams as RMSpropHparams
from composer.optim.optimizer_hparams import SGDHparams as SGDHparams
from composer.optim.scheduler import ComposerScheduler as ComposerScheduler
from composer.optim.scheduler import ConstantScheduler as ConstantScheduler
from composer.optim.scheduler import CosineAnnealingScheduler as CosineAnnealingScheduler
from composer.optim.scheduler import CosineAnnealingWarmRestartsScheduler as CosineAnnealingWarmRestartsScheduler
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler as CosineAnnealingWithWarmupScheduler
from composer.optim.scheduler import ExponentialScheduler as ExponentialScheduler
from composer.optim.scheduler import LinearScheduler as LinearScheduler
from composer.optim.scheduler import LinearWithWarmupScheduler as LinearWithWarmupScheduler
from composer.optim.scheduler import MultiStepScheduler as MultiStepScheduler
from composer.optim.scheduler import MultiStepWithWarmupScheduler as MultiStepWithWarmupScheduler
from composer.optim.scheduler import PolynomialScheduler as PolynomialScheduler
from composer.optim.scheduler import StepScheduler as StepScheduler
from composer.optim.scheduler_hparams import ConstantSchedulerHparams as ConstantSchedulerHparams
from composer.optim.scheduler_hparams import CosineAnnealingSchedulerHparams as CosineAnnealingSchedulerHparams
from composer.optim.scheduler_hparams import \
    CosineAnnealingWarmRestartsSchedulerHparams as CosineAnnealingWarmRestartsSchedulerHparams
from composer.optim.scheduler_hparams import \
    CosineAnnealingWithWarmupSchedulerHparams as CosineAnnealingWithWarmupSchedulerHparams
from composer.optim.scheduler_hparams import ExponentialSchedulerHparams as ExponentialSchedulerHparams
from composer.optim.scheduler_hparams import LinearSchedulerHparams as LinearSchedulerHparams
from composer.optim.scheduler_hparams import LinearWithWarmupSchedulerHparams as LinearWithWarmupSchedulerHparams
from composer.optim.scheduler_hparams import MultiStepSchedulerHparams as MultiStepSchedulerHparams
from composer.optim.scheduler_hparams import MultiStepWithWarmupSchedulerHparams as MultiStepWithWarmupSchedulerHparams
from composer.optim.scheduler_hparams import PolynomialSchedulerHparams as PolynomialSchedulerHparams
from composer.optim.scheduler_hparams import SchedulerHparams as SchedulerHparams
from composer.optim.scheduler_hparams import StepSchedulerHparams as StepSchedulerHparams
