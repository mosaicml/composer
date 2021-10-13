# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Dict, Type
from unittest import mock

import pytest
import torch
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR

from composer.core.types import ModelParameters
from composer.optim.pytorch_future import WarmUpLR
from composer.optim.scheduler import (ComposedScheduler, ConstantLRHparams, CosineAnnealingLRHparams,
                                      CosineAnnealingWarmRestartsHparams, ExponentialLRHparams, MultiStepLRHparams,
                                      SchedulerHparams, StepLRHparams, WarmUpLRHparams)
from composer.trainer.trainer_hparams import scheduler_registry

# for testing, we provide values for required hparams fields
schedulers: Dict[Type[SchedulerHparams], SchedulerHparams] = {
    StepLRHparams: StepLRHparams(step_size="5ep",),
    MultiStepLRHparams: MultiStepLRHparams(milestones=["5ep", "10ep"],),
    ExponentialLRHparams: ExponentialLRHparams(gamma=0.5,),
    CosineAnnealingLRHparams: CosineAnnealingLRHparams(T_max="1000ep",),
    CosineAnnealingWarmRestartsHparams: CosineAnnealingWarmRestartsHparams(T_0="1000ep",),
    WarmUpLRHparams: WarmUpLRHparams(),
    ConstantLRHparams: ConstantLRHparams()
}

time_field: Dict[Type[SchedulerHparams], str] = {
    StepLRHparams: 'step_size',
    MultiStepLRHparams: '',
    ExponentialLRHparams: '',
    CosineAnnealingLRHparams: 'T_max',
    CosineAnnealingWarmRestartsHparams: 'T_0',
    WarmUpLRHparams: 'warmup_iters',
    ConstantLRHparams: ''
}

EXPECTED_RESULTS_TIME_CONVERSION = {
    '33ep12ba': {
        'steps': 3312,
        'epochs': 33
    },
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
    }
}


@pytest.fixture
def dummy_parameters() -> ModelParameters:
    net = torch.nn.Sequential(torch.nn.Linear(5, 2), torch.nn.ReLU())
    return net.parameters()


@pytest.fixture
def dummy_optimizer(dummy_parameters) -> torch.optim.Optimizer:
    return torch.optim.SGD(dummy_parameters, 0.1)


@pytest.mark.parametrize("scheduler_name", scheduler_registry.keys())
class TestSchedulerInit():

    def test_scheduler_initialization(self, scheduler_name, dummy_optimizer):

        # create the scheduler hparams object
        obj: Type[SchedulerHparams] = scheduler_registry[scheduler_name]
        scheduler_hparams = schedulers[obj]

        # create the scheduler object using the hparams
        scheduler, interval = scheduler_hparams.initialize_object(dummy_optimizer, steps_per_epoch=1)
        assert isinstance(scheduler, scheduler_hparams.scheduler_object)  # type: ignore
        assert interval == scheduler_hparams.interval  # type: ignore

    @pytest.mark.parametrize('timestrings', EXPECTED_RESULTS_TIME_CONVERSION.keys())
    @pytest.mark.parametrize('interval', ['steps', 'epochs'])
    def test_scheduler_time_conversion(self, scheduler_name, dummy_optimizer, timestrings, interval):
        expected = EXPECTED_RESULTS_TIME_CONVERSION[timestrings][interval]
        obj: Type[SchedulerHparams] = scheduler_registry[scheduler_name]

        if time_field[obj]:
            scheduler_hparams = schedulers[obj]
            with mock.patch.object(scheduler_hparams, time_field[obj], timestrings), \
                mock.patch.object(scheduler_hparams, 'interval', interval):

                scheduler, interval = scheduler_hparams.initialize_object(dummy_optimizer, steps_per_epoch=100)

                assert getattr(scheduler, time_field[obj]) == expected


@pytest.fixture
def optimizer(dummy_model):
    return torch.optim.SGD(dummy_model.parameters(), lr=1)


class TestComposedScheduler():

    def _test(self, scheduler, targets, epochs, optimizer, interval='epoch'):
        for epoch in range(epochs):
            for param_group, target in zip(optimizer.param_groups, targets):
                torch.testing.assert_allclose(target[epoch], param_group['lr'])
            optimizer.step()
            scheduler.step(interval)

    def test_composed(self, optimizer):
        epochs = 9
        targets = [[1 * 0.2 for x in range(4)] + [1 * 0.9**x for x in range(7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0.2, warmup_iters=4, warmup_method="constant")
        ]
        schedulers = [(s, 'epoch') for s in schedulers]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear(self, optimizer):
        epochs = 9
        targets = [[1 * 0.5 + (x/4 * 0.5) for x in range(4)] + [1 * 0.9**x for x in range(2)] + \
                   [1 * 0.9**x for x in range(2, 7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0.5, warmup_iters=4, warmup_method="linear")
        ]
        schedulers = [(s, 'epoch') for s in schedulers]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear2(self, optimizer):
        epochs = 9
        targets = [[1 * 0.5 + (x/4 * 0.5) for x in range(4)] + \
                   [1 * 0.9**x for x in range(2)] + [1 * 0.9**x * 0.1 for x in range(2, 7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            MultiStepLR(optimizer, milestones=[6], gamma=0.1),
            WarmUpLR(optimizer, warmup_factor=0.5, warmup_iters=4, warmup_method="linear")
        ]
        schedulers = [(s, 'epoch') for s in schedulers]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear_from_zero(self, optimizer):
        epochs = 9
        targets = [[1 * 0.0 + (x / 4 * 1.0) for x in range(4)] + [1 * 0.9**x for x in range(7)]]
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear")
        ]
        schedulers = [(s, 'epoch') for s in schedulers]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer)

    def test_composed_linear_from_zero_step(self, optimizer):
        epochs = 9
        targets = [[x / 4 for x in range(4)] + [1.0 for _ in range(7)]]
        schedulers = [
            (ExponentialLR(optimizer, gamma=0.9), 'epoch'),  # should never trigger
            (WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear"), 'batch')
        ]
        scheduler = ComposedScheduler(schedulers)
        self._test(scheduler, targets, epochs, optimizer, interval='batch')

    @pytest.mark.xfail
    def test_validate_compose_multistep(self, optimizer):
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear"),
            MultiStepLR(optimizer, milestones=[3], gamma=0.1)
        ]
        schedulers = [(s, 'epoch') for s in schedulers]

        with pytest.raises(ValueError):
            ComposedScheduler(schedulers)

    @pytest.mark.xfail
    def test_validate_compose_step(self, optimizer):
        schedulers = [
            ExponentialLR(optimizer, gamma=0.9),
            WarmUpLR(optimizer, warmup_factor=0, warmup_iters=4, warmup_method="linear"),
            StepLR(optimizer, step_size=2, gamma=0.1)
        ]
        schedulers = [(s, 'epoch') for s in schedulers]

        with pytest.raises(ValueError):
            ComposedScheduler(schedulers)
