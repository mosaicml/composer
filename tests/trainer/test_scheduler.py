# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Dict, List, Type, Union
from unittest import mock

import pytest
import torch
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR

from composer.core.types import Optimizer, Scheduler
from composer.optim.scheduler import (ConstantLRHparams, CosineAnnealingLRHparams, CosineAnnealingWarmRestartsHparams,
                                      ExponentialLRHparams, LinearLRHparams, MultiStepLRHparams, PolynomialLRHparams,
                                      SchedulerHparams, StepLRHparams)
from composer.trainer.trainer_hparams import scheduler_registry

# for testing, we provide values for required hparams fields
MAX_EPOCHS = 1000
schedulers: Dict[Type[SchedulerHparams], SchedulerHparams] = {
    StepLRHparams: StepLRHparams(step_size="5ep",),
    MultiStepLRHparams: MultiStepLRHparams(milestones=["5ep", "10ep"],),
    ExponentialLRHparams: ExponentialLRHparams(gamma=0.5,),
    CosineAnnealingLRHparams: CosineAnnealingLRHparams(T_max=f"{MAX_EPOCHS}ep",),
    LinearLRHparams: LinearLRHparams(total_time=f"{MAX_EPOCHS}ep",),
    CosineAnnealingWarmRestartsHparams: CosineAnnealingWarmRestartsHparams(T_0=f"{MAX_EPOCHS}ep",),
    ConstantLRHparams: ConstantLRHparams(),
    PolynomialLRHparams: PolynomialLRHparams(T_max="100ep", power=0.9)
}

time_field: Dict[Type[SchedulerHparams], str] = {
    StepLRHparams: 'step_size',
    MultiStepLRHparams: '',
    ExponentialLRHparams: '',
    CosineAnnealingLRHparams: 'T_max',
    LinearLRHparams: 'total_iters',
    CosineAnnealingWarmRestartsHparams: 'T_0',
    ConstantLRHparams: '',
    PolynomialLRHparams: 'T_max'
}

EXPECTED_RESULTS_TIME_CONVERSION = {
    '17ep': {
        'steps': 1700,
        'epochs': 17
    },
    '5050ba': {
        'steps': 5050,
        'epochs': 50
    },
    42: {
        'steps': 42,
        'epochs': 42
    },
    '0.05dur': {
        'steps': 5000,
        'epochs': 50,
    },
    '0.95dur': {
        'steps': 95000,
        'epochs': 950,
    }
}

TIME_HPARAMS = {
    '33ep12ba': {
        'max_epochs': MAX_EPOCHS,
        'steps_per_epoch': 100,
    },
    '17ep': {
        'max_epochs': MAX_EPOCHS,
        'steps_per_epoch': 100,
    },
    '5050ba': {
        'max_epochs': MAX_EPOCHS,
        'steps_per_epoch': 100,
    },
    42: {
        'max_epochs': MAX_EPOCHS,
        'steps_per_epoch': 100,
    },
    '0.05dur': {
        'max_epochs': MAX_EPOCHS,
        'steps_per_epoch': 100,
    },
    '0.95dur': {
        'max_epochs': MAX_EPOCHS,
        'steps_per_epoch': 100,
    }
}


@pytest.mark.parametrize("scheduler_name", scheduler_registry.keys())
class TestSchedulerInit():

    def test_scheduler_initialization(self, scheduler_name: str, dummy_optimizer: Optimizer):

        # create the scheduler hparams object
        obj: Type[SchedulerHparams] = scheduler_registry[scheduler_name]
        scheduler_hparams = schedulers[obj]

        # create the scheduler object using the hparams
        scheduler = scheduler_hparams.initialize_object()
        assert isinstance(scheduler, scheduler_hparams.scheduler_object)  # type: ignore

    @pytest.mark.parametrize('timestrings', EXPECTED_RESULTS_TIME_CONVERSION.keys())
    @pytest.mark.parametrize('interval', ['steps', 'epochs'])
    def test_scheduler_time_conversion(self, scheduler_name: str, dummy_optimizer: Optimizer,
                                       timestrings: Union[str, int], interval: str):
        expected = EXPECTED_RESULTS_TIME_CONVERSION[timestrings][interval]
        obj: Type[SchedulerHparams] = scheduler_registry[scheduler_name]
        steps_per_epoch = TIME_HPARAMS[timestrings]['steps_per_epoch']
        max_epochs = TIME_HPARAMS[timestrings]['max_epochs']

        if time_field[obj]:
            scheduler_hparams = schedulers[obj]
            with mock.patch.object(scheduler_hparams, time_field[obj], timestrings), \
                mock.patch.object(scheduler_hparams, 'interval', interval):

                scheduler = scheduler_hparams.initialize_object()

                assert getattr(scheduler, time_field[obj]) == expected


@pytest.fixture
def optimizer(dummy_model: torch.nn.Module):
    return torch.optim.SGD(dummy_model.parameters(), lr=1)


class TestComposedScheduler():

    def _test(self,
              scheduler: Scheduler,
              targets: List[List[float]],
              epochs: int,
              optimizer: Optimizer,
              interval: str = 'epoch'):
        for epoch in range(epochs):
            for param_group, target in zip(optimizer.param_groups, targets):
                torch.testing.assert_allclose(target[epoch], param_group['lr'])
            optimizer.step()
            scheduler.step(interval)  # type: ignore

    def test_composed(self, optimizer: Optimizer):
        epochs = 9
        targets = [[1 * 0.2 for _ in range(4)] + [1 * 0.9**x for x in range(7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0.2, warmup_iters=4, warmup_method="constant")
        ]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear(self, optimizer: Optimizer):
        epochs = 9
        targets = [[1 * 0.5 + (x/4 * 0.5) for x in range(4)] + [1 * 0.9**x for x in range(2)] + \
                   [1 * 0.9**x for x in range(2, 7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0.5, warmup_iters=4, warmup_method="linear")
        ]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear2(self, optimizer: Optimizer):
        epochs = 9
        targets = [[1 * 0.5 + (x/4 * 0.5) for x in range(4)] + \
                   [1 * 0.9**x for x in range(2)] + [1 * 0.9**x * 0.1 for x in range(2, 7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            MultiStepLR(optimizer, milestones=[6], gamma=0.1),
            WarmUpLR(optimizer, warmup_factor=0.5, warmup_iters=4, warmup_method="linear")
        ]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear_from_zero(self, optimizer: Optimizer):
        epochs = 9
        targets = [[1 * 0.0 + (x / 4 * 1.0) for x in range(4)] + [1 * 0.9**x for x in range(7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear")
        ]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear_from_zero_step(self, optimizer: Optimizer):
        epochs = 9
        targets = [[x / 4 for x in range(4)] + [1.0 for _ in range(7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear"),
        ]
        schedulers[0].interval = 'epoch'  # should never trigger
        schedulers[1].interval = 'batch'
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer, interval='batch')

    @pytest.mark.xfail
    def test_validate_compose_multistep(self, optimizer: Optimizer):
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear"),
            MultiStepLR(optimizer, milestones=[3], gamma=0.1)
        ]

        with pytest.raises(ValueError):
            ComposedScheduler(schedulers)

    @pytest.mark.xfail
    def test_validate_compose_step(self, optimizer: Optimizer):
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear"),
            StepLR(optimizer, step_size=2, gamma=0.1)
        ]

        with pytest.raises(ValueError):
            ComposedScheduler(schedulers)
