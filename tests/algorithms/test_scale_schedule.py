# Copyright 2021 MosaicML. All Rights Reserved.

from collections import Counter

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from composer.algorithms import ScaleSchedule
from composer.algorithms.scale_schedule import scale_scheduler
from composer.core.event import Event
from composer.core.logging import Logger
from composer.core.state import State
from composer.core.types import Optimizer, Scheduler
from composer.optim.pytorch_future import WarmUpLR


@pytest.fixture
def optimizer(dummy_model: torch.nn.Module):
    return torch.optim.SGD(dummy_model.parameters(), lr=1)


def flatten(lst: list):
    return [x for sublst in lst for x in sublst]


@pytest.mark.parametrize('ssr', [0.5, 0.75, 1.0])
class TestScaleSchedule():

    @staticmethod
    def _test(targets, scheduler: Scheduler, epochs: int, optimizer: Optimizer, ssr: float):
        scale_scheduler(scheduler, ssr)
        for epoch in range(epochs):
            for param_group in optimizer.param_groups:
                print(f"target: {targets[epoch]}, actual: {param_group['lr']}")
                torch.testing.assert_allclose(targets[epoch], param_group['lr'])
            scheduler.step()

    def test_scale_schedule_step_lr(self, optimizer: Optimizer, ssr: float):
        epochs = int(9 * ssr)
        step_size = int(3 * ssr)
        gamma = 0.1
        targets = flatten([[1.0 * (gamma**n)] * step_size for n in range(30)])
        targets = targets[:epochs]

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        self._test(targets, scheduler, epochs, optimizer, ssr)

    def test_scale_schedule_multistep_lr(self, optimizer: Optimizer, ssr: float):
        epochs = int(9 * ssr)
        milestones = np.diff([0, int(2 * ssr), int(7 * ssr), epochs])
        gamma = 0.1
        targets = flatten([[1.0 * (gamma**n)] * ms for n, ms in enumerate(milestones)])
        targets = targets[:epochs]

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 7], gamma)
        self._test(targets, scheduler, epochs, optimizer, ssr)

    def test_scale_schedule_exponential(self, optimizer: Optimizer, ssr: float):
        epochs = int(9 * ssr)
        targets = [1.0 * 0.1**(x / ssr) for x in range(epochs)]
        scheduler = ExponentialLR(optimizer, gamma=0.1)
        self._test(targets, scheduler, epochs, optimizer, ssr)

    @pytest.mark.xfail
    def test_scale_schedule_cosine(self, optimizer: Optimizer, ssr: float):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_scale_schedule_cosine_warm_restarts(self, optimizer: Optimizer, ssr: float):
        raise NotImplementedError

    def test_scale_schedule_warmup(self, optimizer: Optimizer, ssr: float):
        targets = [0.5] * 4 + [1.0] * 5  # no effect
        scheduler = WarmUpLR(optimizer, warmup_factor=0.5, warmup_iters=4, warmup_method='constant', interval='step')
        epochs = int(9 * ssr)
        targets = targets[:epochs]
        self._test(targets, scheduler, epochs, optimizer, ssr)


@pytest.mark.parametrize('ssr', [0.5, 0.75, 1.0])
class TestScaleScheduleAlgorithm():

    def test_epochs_scaled(self, dummy_state: State, optimizer: Optimizer, ssr: float, noop_dummy_logger: Logger):

        scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
        dummy_state.schedulers = scheduler
        dummy_state.max_duration = "10ep"
        algorithm = ScaleSchedule(ratio=ssr)
        algorithm.apply(Event.TRAINING_START, dummy_state, noop_dummy_logger)
        assert dummy_state.max_epochs == int(10 * ssr)
        assert scheduler.milestones == Counter([int(30 * ssr), int(50 * ssr)])  # type: ignore

    @pytest.mark.xfail
    def test_scale_schedule_compose1(self, optimizer: Optimizer, ssr: float):
        return NotImplementedError

    @pytest.mark.xfail
    def test_scale_schedule_compose2(self, optimizer: Optimizer, ssr: float):
        return NotImplementedError


def test_epochs_validate_zero_epochs(dummy_state: State, noop_dummy_logger: Logger):
    algorithm = ScaleSchedule(ratio=0.01)
    dummy_state.max_duration = "10ep"
    dummy_state.schedulers = tuple()
    with pytest.raises(ValueError):
        algorithm.apply(Event.TRAINING_START, dummy_state, noop_dummy_logger)
