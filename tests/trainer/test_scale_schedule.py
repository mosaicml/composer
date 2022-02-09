# Copyright 2021 MosaicML. All Rights Reserved.

from collections import Counter

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import ExponentialLR

from composer.algorithms import ScaleScheduleHparams
from composer.core.types import Optimizer, Scheduler
from composer.optim.pytorch_future import WarmUpLR
from composer.optim.scheduler import MultiStepLRHparams
from composer.trainer import TrainerHparams
from composer.trainer.scale_schedule import scale_scheduler
from tests.common import SimpleModel


@pytest.fixture
def optimizer():
    return torch.optim.SGD(SimpleModel().parameters(), lr=1)


def flatten(lst: list):
    return [x for sublst in lst for x in sublst]


@pytest.mark.parametrize('ssr', [0.5, 0.75, 1.0])
class TestScaleSchedule():

    @staticmethod
    def _test(targets, scheduler: Scheduler, epochs: int, optimizer: Optimizer, ssr: float):
        scale_scheduler(scheduler, ssr)
        for epoch in range(epochs):
            for param_group in optimizer.param_groups:
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
@pytest.mark.parametrize('use_algorithm', [False, True])
class TestScaleScheduleTrainer():

    def test_epochs_scaled(self, ssr: float, use_algorithm: bool, composer_trainer_hparams: TrainerHparams):

        composer_trainer_hparams.max_duration = '10ep'
        composer_trainer_hparams.schedulers = [MultiStepLRHparams(milestones=[30, 50], gamma=0.1)]

        if use_algorithm:
            composer_trainer_hparams.algorithms = [ScaleScheduleHparams(ratio=ssr)]
        else:
            composer_trainer_hparams.scale_schedule_ratio = ssr
        trainer = composer_trainer_hparams.initialize_object()

        assert trainer.state.max_epochs == int(10 * ssr)
        scheduler = trainer.state.schedulers[0].schedulers[0]  # type: ignore
        assert scheduler.milestones == Counter([int(30 * ssr), int(50 * ssr)])  # type: ignore
