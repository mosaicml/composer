# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict

import pytest

from composer.optim.scheduler import (ComposerScheduler, ConstantWithWarmupScheduler,
                                      CosineAnnealingWarmRestartsScheduler, CosineAnnealingWithWarmupScheduler,
                                      ExponentialScheduler, LinearWithWarmupScheduler, MultiStepScheduler,
                                      MultiStepWithWarmupScheduler, PolynomialScheduler, PolynomialWithWarmupScheduler,
                                      StepScheduler)
from composer.optim.scheduler_hparams_registry import scheduler_registry
from tests.common.hparams import construct_from_yaml

# Cannot query the module and use an isinstance check because schedulers have no base class -- they're just functions
# that return functions. Instead, using the registry
scheduler_classes = scheduler_registry.values()

scheduler_settings: Dict[Callable[..., ComposerScheduler], Dict[str, Any]] = {
    StepScheduler: {
        'step_size': 1,
    },
    MultiStepScheduler: {
        'milestones': [0],
    },
    ExponentialScheduler: {
        'gamma': 1,
    },
    CosineAnnealingWarmRestartsScheduler: {
        't_0': 0,
    },
    PolynomialScheduler: {
        'power': 0.1,
    },
    PolynomialWithWarmupScheduler: {
        't_warmup': '0ep',
    },
    MultiStepWithWarmupScheduler: {
        'milestones': [0],
        't_warmup': '0ep',
    },
    ConstantWithWarmupScheduler: {
        't_warmup': '0ep',
    },
    LinearWithWarmupScheduler: {
        't_warmup': '0ep',
    },
    CosineAnnealingWithWarmupScheduler: {
        't_warmup': '0ep',
    },
}


@pytest.mark.parametrize('scheduler_cls', scheduler_classes)
class TestSchedulers:

    def test_scheduler_is_constructable(self, scheduler_cls: Callable[..., ComposerScheduler]):
        kwargs = scheduler_settings.get(scheduler_cls, {})
        scheduler = scheduler_cls(**kwargs)
        assert callable(scheduler)

    def test_scheduler_is_constructable_from_hparams(self, scheduler_cls: Callable[..., ComposerScheduler]):
        kwargs = scheduler_settings.get(scheduler_cls, {})
        construct_from_yaml(scheduler_cls, kwargs)
