# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np
import pytest
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from composer.core import Callback, State, TimeUnit
from composer.core.types import PyTorchScheduler
from composer.loggers.logger import Logger
from composer.optim import MultiStepScheduler
from composer.optim.optimizer_hparams_registry import SGDHparams
from composer.trainer._scale_schedule import scale_pytorch_scheduler
from composer.trainer.trainer_hparams import TrainerHparams
from tests.common.models import SimpleModel


@pytest.fixture
def optimizer():
    return torch.optim.SGD(SimpleModel().parameters(), lr=1.0)


def flatten(lst: list):
    return [x for sublst in lst for x in sublst]


@pytest.mark.parametrize('ssr', [0.5, 0.75, 1.0])
@pytest.mark.filterwarnings(r'ignore:.*Detected call of \`lr_schedule.*:UserWarning')
class TestScaleSchedule():

    @staticmethod
    def _test(targets: List[float], scheduler: PyTorchScheduler, epochs: int, optimizer: Optimizer, ssr: float):
        scale_pytorch_scheduler(scheduler, ssr)
        for epoch in range(epochs):
            for param_group in optimizer.param_groups:
                torch.testing.assert_close(targets[epoch], param_group['lr'])
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


class CheckScaleSchedule(Callback):

    def __init__(self, ssr: float) -> None:
        self.ssr = ssr

    def fit_start(self, state: State, logger: Logger) -> None:
        scheduler = state.schedulers[0]

        test_steps = [int(20 * self.ssr), int(40 * self.ssr), int(60 * self.ssr)]
        target_lrs = [1.0, 0.1, 0.01]
        current_step = 0
        for test_step, target_lr in zip(test_steps, target_lrs):

            while current_step < test_step:
                state.timestamp = state.timestamp.to_next_batch()
                current_step += 1

            scheduler.step()

            assert scheduler.get_last_lr()[0] == pytest.approx(target_lr)


@pytest.mark.parametrize('ssr', [0.5, 0.75, 1.0])
class TestScaleScheduleTrainer():

    @pytest.mark.filterwarnings(r'ignore:.*Detected call of \`lr_schedule.*:UserWarning')
    def test_epochs_scaled(
        self,
        ssr: float,
        composer_trainer_hparams: TrainerHparams,
    ):

        composer_trainer_hparams.optimizers = SGDHparams(lr=1.0)
        composer_trainer_hparams.max_duration = '10ep'
        composer_trainer_hparams.schedulers = [MultiStepScheduler(milestones=['30ba', '50ba'], gamma=0.1)]

        composer_trainer_hparams.scale_schedule_ratio = ssr
        trainer = composer_trainer_hparams.initialize_object()

        trainer = composer_trainer_hparams.initialize_object()
        trainer.state.callbacks.append(CheckScaleSchedule(ssr))

        assert trainer.state.max_duration is not None
        assert trainer.state.max_duration.unit == TimeUnit.EPOCH
        assert trainer.state.max_duration.value == int(10 * ssr)

        trainer.fit()
