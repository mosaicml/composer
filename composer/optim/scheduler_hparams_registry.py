# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters for schedulers."""

from typing import Dict, Type, Union

import yahp as hp

from composer.optim.scheduler import (ComposerScheduler, ConstantScheduler, ConstantWithWarmupScheduler,
                                      CosineAnnealingScheduler, CosineAnnealingWarmRestartsScheduler,
                                      CosineAnnealingWithWarmupScheduler, ExponentialScheduler, LinearScheduler,
                                      LinearWithWarmupScheduler, MultiStepScheduler, MultiStepWithWarmupScheduler,
                                      PolynomialScheduler, PolynomialWithWarmupScheduler, StepScheduler)

__all__ = ['scheduler_registry']

scheduler_registry: Dict[str, Union[Type[ComposerScheduler], Type[hp.Hparams]]] = {
    'step': StepScheduler,
    'multistep': MultiStepScheduler,
    'exponential': ExponentialScheduler,
    'linear_decay': LinearScheduler,
    'cosine_decay': CosineAnnealingScheduler,
    'cosine_warmrestart': CosineAnnealingWarmRestartsScheduler,
    'constant': ConstantScheduler,
    'polynomial': PolynomialScheduler,
    'polynomial_with_warmup': PolynomialWithWarmupScheduler,
    'multistep_with_warmup': MultiStepWithWarmupScheduler,
    'constant_with_warmup': ConstantWithWarmupScheduler,
    'linear_decay_with_warmup': LinearWithWarmupScheduler,
    'cosine_decay_with_warmup': CosineAnnealingWithWarmupScheduler,
}
