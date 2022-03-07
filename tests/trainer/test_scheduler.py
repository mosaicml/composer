# Copyright 2021 MosaicML. All Rights Reserved.

import contextlib
from typing import List, Optional, Type, cast

import pytest

from composer.core import State, Time
from composer.core.time import TimeUnit
from composer.core.types import DataLoader, Model
from composer.models.base import ComposerModel
from composer.optim.scheduler import (ComposerScheduler, CosineAnnealingScheduler, CosineAnnealingWarmRestartsScheduler,
                                      CosineAnnealingWithWarmupScheduler, ExponentialScheduler, LinearScheduler,
                                      LinearWithWarmupScheduler, MultiStepScheduler, MultiStepWithWarmupScheduler,
                                      PolynomialScheduler, StepScheduler)
from composer.trainer.trainer import Trainer

MAX_DURATION = '1000ep'
STEPS_PER_EPOCH = 1000


@pytest.fixture
def dummy_schedulers_state(dummy_model: Model, dummy_train_dataloader: DataLoader):
    return State(
        model=dummy_model,
        train_dataloader=dummy_train_dataloader,
        max_duration=MAX_DURATION,
        steps_per_epoch=STEPS_PER_EPOCH,
    )


@pytest.mark.parametrize("scheduler,ssr,test_times,expected_lrs", [
    pytest.param(StepScheduler(step_size='10ba'), 1.0, ['5ba', '15ba', '35ba'], [1.0, 0.1, 0.001]),
    pytest.param(StepScheduler(step_size='0.002dur', gamma=0.8), 1.0, ['1000ba', '3000ba', '7000ba'],
                 [1.0, 0.8, 0.512]),
    pytest.param(StepScheduler(step_size='1ep', gamma=0.5), 1.0, ['500ba', '1500ba', '3500ba'], [1.0, 0.5, 0.125]),
    pytest.param(StepScheduler(step_size='10ba', gamma=0.5), 0.5, ['3ba', '8ba', '18ba'], [1.0, 0.5, 0.125]),
    pytest.param(MultiStepScheduler(milestones=cast(List, ['10ba', '30ba', '70ba'])), 1.0,
                 ['5ba', '20ba', '50ba', '100ba'], [1.0, 0.1, 0.01, 0.001]),
    pytest.param(MultiStepScheduler(milestones=cast(List, ['100ba', '1ep', '0.01dur']), gamma=0.5), 1.0,
                 ['50ba', '500ba', '5000ba', '50000ba'], [1.0, 0.5, 0.25, 0.125]),
    pytest.param(MultiStepScheduler(milestones=cast(List, ['100ba', '1ep', '0.01dur']), gamma=0.5), 4.0,
                 ['200ba', '2000ba', '20000ba', '200000ba'], [1.0, 0.5, 0.25, 0.125]),
    pytest.param(LinearScheduler(), 1.0, ['100000ba', '200000ba', '400000ba'], [0.9, 0.8, 0.6]),
    pytest.param(LinearScheduler(start_factor=0.0, end_factor=2.0), 1.0, ['100000ba', '200000ba', '400000ba'],
                 [0.2, 0.4, 0.8]),
    pytest.param(LinearScheduler(start_factor=0.0, end_factor=1.0, total_time='0.25dur'), 1.0,
                 ['100000ba', '200000ba', '400000ba'], [0.4, 0.8, 1.0]),
    pytest.param(LinearScheduler(start_factor=0.0, end_factor=1.0, total_time='0.25dur'), 2.0,
                 ['100000ba', '200000ba', '400000ba'], [0.2, 0.4, 0.8]),
    pytest.param(ExponentialScheduler(gamma=0.5), 1.0, ['1ep', '2ep', '4ep'], [0.5, 0.25, 0.0625]),
    pytest.param(ExponentialScheduler(gamma=0.5), 2.0, ['2ep', '4ep', '8ep'], [0.5, 0.25, 0.0625]),
    pytest.param(CosineAnnealingScheduler(), 1.0, ['0ba', '333333ba', '500000ba', '666667ba', '1000000ba'],
                 [1.0, 0.75, 0.5, 0.25, 0.0]),
    pytest.param(CosineAnnealingScheduler(t_max='30ba', min_factor=0.5), 1.0,
                 ['0ba', '10ba', '15ba', '20ba', '30ba', '50ba'], [1.0, 0.875, 0.75, 0.625, 0.5, 0.5]),
    pytest.param(CosineAnnealingScheduler(t_max='30ba', min_factor=0.5), 0.2,
                 ['0ba', '2ba', '3ba', '4ba', '6ba', '10ba'], [1.0, 0.875, 0.75, 0.625, 0.5, 0.5]),
    pytest.param(CosineAnnealingWarmRestartsScheduler(t_0='30ba'), 1.0, ['0ba', '10ba', '15ba', '20ba', '30ba', '40ba'],
                 [1.0, 0.75, 0.5, 0.25, 1.0, 0.75]),
    pytest.param(CosineAnnealingWarmRestartsScheduler(t_0='0.003dur', t_mult=1.5), 1.0,
                 ['0ba', '1000ba', '3000ba', '4500ba', '7500ba', '14250ba'], [1.0, 0.75, 1.0, 0.75, 1.0, 1.0]),
    pytest.param(CosineAnnealingWarmRestartsScheduler(t_0='30ep', t_mult=2.0, min_factor=0.5), 0.5,
                 ['0ba', '5000ba', '15000ba', '25000ba'], [1.0, 0.875, 1.0, 0.875]),
    pytest.param(PolynomialScheduler(power=2.0), 1.0, ['0ba', '100000ba', '200000ba', '500000ba'],
                 [1.0, 0.81, 0.64, 0.25]),
    pytest.param(PolynomialScheduler(power=2.0, t_max='100ba', min_factor=0.5), 1.0, ['0ba', '10ba', '20ba', '50ba'],
                 [1.0, 0.905, 0.82, 0.625]),
    pytest.param(PolynomialScheduler(power=2.0, t_max='100ba', min_factor=0.5), 0.5, ['0ba', '10ba', '20ba', '50ba'],
                 [1.0, 0.82, 0.68, 0.5]),
    pytest.param(MultiStepWithWarmupScheduler(warmup_time='10ba', milestones=cast(List, ['20ba', '40ba'])), 1.0,
                 ['0ba', '5ba', '15ba', '25ba', '45ba'], [0.0, 0.5, 1.0, 0.1, 0.01]),
    pytest.param(MultiStepWithWarmupScheduler(warmup_time='10ba', milestones=cast(List, ['2ep', '4ep']), gamma=0.5),
                 0.5, ['0ba', '5ba', '15ba', '1500ba', '2500ba'], [0.0, 0.5, 1.0, 0.5, 0.25]),
    pytest.param(LinearWithWarmupScheduler(warmup_time='500ep'), 1.0,
                 ['0ba', '250000ba', '500000ba', '750000ba', '1000000ba'], [0.0, 0.5, 1.0, 0.5, 0.0]),
    pytest.param(LinearWithWarmupScheduler(warmup_time='500ep', start_factor=3.0, end_factor=2.0,
                                           total_time='1002ep'), 0.5,
                 ['0ba', '250000ba', '500000ba', '500500ba', '501000ba', '502000ba'], [0.0, 1.5, 3.0, 2.5, 2.0, 2.0]),
    pytest.param(LinearWithWarmupScheduler(warmup_time='0.0005dur'), 1.0, ['0ba', '250ba', '500ba', '499750ba'],
                 [0.0, 0.5, 1.0, 0.5]),
    pytest.param(CosineAnnealingWithWarmupScheduler(warmup_time='0.9dur'), 1.0,
                 ['0ba', '450000ba', '900000ba', '933333ba', '950000ba', '1000000ba'], [0.0, 0.5, 1.0, 0.75, 0.5, 0.0]),
    pytest.param(CosineAnnealingWithWarmupScheduler(warmup_time='0.9dur', min_factor=0.5), 0.01,
                 ['0ba', '4500ba', '9000ba', '9333ba', '9500ba', '10000ba'], [0.0, 0.5, 1.0, 0.875, 0.75, 0.5]),
])
def test_scheduler_init(scheduler: ComposerScheduler, ssr: float, test_times: List[str], expected_lrs: List[float],
                        dummy_schedulers_state: State):

    state = dummy_schedulers_state
    state._max_duration = Time(value=int(state.max_duration.value * ssr), unit=state.max_duration.unit)
    for test_time, expected_lr in zip(test_times, expected_lrs):
        parsed_time = Time.from_timestring(test_time)
        assert parsed_time.unit in [TimeUnit.EPOCH, TimeUnit.BATCH]
        if parsed_time.unit == TimeUnit.EPOCH:
            state.timer._epoch = parsed_time
            state.timer._batch = Time(int(state.steps_per_epoch * state.timer._epoch.value), TimeUnit.BATCH)
        else:
            state.timer._batch = parsed_time
            state.timer._epoch = Time(int(state.timer._batch.value / state.steps_per_epoch), TimeUnit.EPOCH)

        lr = scheduler(state, ssr)
        assert lr == pytest.approx(expected_lr, abs=1e-3)


@pytest.mark.parametrize(
    "scheduler,ssr,should_raise",
    [
        (StepScheduler(step_size='2ba'), 1.0, None),
        (StepScheduler(step_size='0.2dur', gamma=0.8), 0.5, None),
        (lambda state, ssr=1.0: 0.01 * ssr, 1.5, None),  # lambda's are also allowed as a ComposerScheduler
        (lambda state: 0.01, 1.0, None),  # if the ssr = 1.0, then the lambda need not take the ssr parameter
        (lambda state: 0.01, 1.5,
         ValueError),  # this should error since the ssr != 1.0 and the lambda doesn't support ssr
    ])
def test_scheduler_trains(scheduler: ComposerScheduler, ssr: float, dummy_model: ComposerModel,
                          dummy_train_dataloader: DataLoader, should_raise: Optional[Type[Exception]]):
    with pytest.raises(should_raise) if should_raise is not None else contextlib.nullcontext():
        trainer = Trainer(
            model=dummy_model,
            train_dataloader=dummy_train_dataloader,
            max_duration='2ep',
            train_subset_num_batches=5,
            scale_schedule_ratio=ssr,
            schedulers=scheduler,
        )
        trainer.fit()
