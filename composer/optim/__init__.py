# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizers and learning rate schedulers.

Composer is compatible with optimizers based off of PyTorch's native :class:`~torch.optim.Optimizer` API, and common
optimizers such
However, where applicable, it is recommended to use the optimizers provided in :mod:`~.decoupled_weight_decay` since
they improve off of their PyTorch equivalents.

PyTorch schedulers can be used with Composer, but this is explicitly discouraged. Instead, it is recommended to use
schedulers based off of Composer's :class:`~.scheduler.ComposerScheduler` API, which allows more flexibility and
configuration in writing schedulers.
"""

from composer.optim.decoupled_weight_decay import DecoupledAdamW, DecoupledSGDW
from composer.optim.scheduler import (ComposerScheduler, ConstantScheduler, ConstantWithWarmupScheduler,
                                      CosineAnnealingScheduler, CosineAnnealingWarmRestartsScheduler,
                                      CosineAnnealingWithWarmupScheduler, ExponentialScheduler, LinearScheduler,
                                      LinearWithWarmupScheduler, MultiStepScheduler, MultiStepWithWarmupScheduler,
                                      PolynomialScheduler, PolynomialWithWarmupScheduler, StepScheduler,
                                      compile_composer_scheduler)

__all__ = [
    'DecoupledAdamW',
    'DecoupledSGDW',
    'ComposerScheduler',
    'ConstantScheduler',
    'ConstantWithWarmupScheduler',
    'CosineAnnealingScheduler',
    'CosineAnnealingWarmRestartsScheduler',
    'CosineAnnealingWithWarmupScheduler',
    'ExponentialScheduler',
    'LinearScheduler',
    'LinearWithWarmupScheduler',
    'MultiStepScheduler',
    'MultiStepWithWarmupScheduler',
    'PolynomialScheduler',
    'PolynomialWithWarmupScheduler',
    'StepScheduler',
    'compile_composer_scheduler',
]
