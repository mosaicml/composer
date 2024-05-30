# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Optional, Type

import pytest
from torch.utils.data import DataLoader

from composer.core import State, Time
from composer.core.time import Timestamp, TimeUnit
from composer.devices import DeviceCPU, DeviceGPU
from composer.optim.scheduler import (
    ComposerScheduler,
    ConstantWithWarmupScheduler,
    CosineAnnealingScheduler,
    CosineAnnealingWarmRestartsScheduler,
    CosineAnnealingWithWarmupScheduler,
    ExponentialScheduler,
    LinearScheduler,
    LinearWithWarmupScheduler,
    MultiStepScheduler,
    MultiStepWithWarmupScheduler,
    PolynomialScheduler,
    PolynomialWithWarmupScheduler,
    StepScheduler,
    _convert_time,
)
from composer.trainer.trainer import Trainer
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel

MAX_DURATION = '1000ep'
STEPS_PER_EPOCH = 1000


@pytest.fixture
def dummy_schedulers_state(rank_zero_seed: int, request: pytest.FixtureRequest):
    device = None
    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None
    state = State(
        model=SimpleModel(),
        run_name='run_name',
        device=device,
        rank_zero_seed=rank_zero_seed,
        max_duration=MAX_DURATION,
    )
    state.set_dataloader([None] * STEPS_PER_EPOCH, 'train')
    return state


@pytest.mark.parametrize(
    'scheduler,ssr,test_times,expected_lrs',
    [
        pytest.param(StepScheduler(step_size='10ba'), 1.0, ['5ba', '15ba', '35ba'], [1.0, 0.1, 0.001]),
        pytest.param(
            StepScheduler(step_size='0.002dur', gamma=0.8),
            1.0,
            ['1000ba', '3000ba', '7000ba'],
            [1.0, 0.8, 0.512],
        ),
        pytest.param(StepScheduler(step_size='1ep', gamma=0.5), 1.0, ['500ba', '1500ba', '3500ba'], [1.0, 0.5, 0.125]),
        pytest.param(StepScheduler(step_size='10ba', gamma=0.5), 0.5, ['3ba', '8ba', '18ba'], [1.0, 0.5, 0.125]),
        pytest.param(
            MultiStepScheduler(milestones=['10ba', '30ba', '70ba']),
            1.0,
            ['5ba', '20ba', '50ba', '100ba'],
            [1.0, 0.1, 0.01, 0.001],
        ),
        pytest.param(
            MultiStepScheduler(milestones=['100ba', '1ep', '0.01dur'], gamma=0.5),
            1.0,
            ['50ba', '500ba', '5000ba', '50000ba'],
            [1.0, 0.5, 0.25, 0.125],
        ),
        pytest.param(
            MultiStepScheduler(milestones=['100ba', '1ep', '0.01dur'], gamma=0.5),
            4.0,
            ['200ba', '2000ba', '20000ba', '200000ba'],
            [1.0, 0.5, 0.25, 0.125],
        ),
        pytest.param(LinearScheduler(), 1.0, ['100000ba', '200000ba', '400000ba'], [0.9, 0.8, 0.6]),
        pytest.param(
            LinearScheduler(alpha_i=0.0, alpha_f=2.0),
            1.0,
            ['100000ba', '200000ba', '400000ba'],
            [0.2, 0.4, 0.8],
        ),
        pytest.param(
            LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max='0.25dur'),
            1.0,
            ['100000ba', '200000ba', '400000ba'],
            [0.4, 0.8, 1.0],
        ),
        pytest.param(
            LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max='0.25dur'),
            2.0,
            ['100000ba', '200000ba', '400000ba'],
            [0.2, 0.4, 0.8],
        ),
        pytest.param(ExponentialScheduler(gamma=0.5), 1.0, ['1ep', '2ep', '4ep'], [0.5, 0.25, 0.0625]),
        pytest.param(ExponentialScheduler(gamma=0.5), 2.0, ['2ep', '4ep', '8ep'], [0.5, 0.25, 0.0625]),
        pytest.param(
            CosineAnnealingScheduler(),
            1.0,
            ['0ba', '333333ba', '500000ba', '666667ba', '1000000ba'],
            [1.0, 0.75, 0.5, 0.25, 0.0],
        ),
        pytest.param(
            CosineAnnealingScheduler(t_max='30ba', alpha_f=0.5),
            1.0,
            ['0ba', '10ba', '15ba', '20ba', '30ba', '50ba'],
            [1.0, 0.875, 0.75, 0.625, 0.5, 0.5],
        ),
        pytest.param(
            CosineAnnealingScheduler(t_max='30ba', alpha_f=0.5),
            0.2,
            ['0ba', '2ba', '3ba', '4ba', '6ba', '10ba'],
            [1.0, 0.875, 0.75, 0.625, 0.5, 0.5],
        ),
        pytest.param(
            CosineAnnealingWarmRestartsScheduler(t_0='30ba'),
            1.0,
            ['0ba', '10ba', '15ba', '20ba', '30ba', '40ba'],
            [1.0, 0.75, 0.5, 0.25, 1.0, 0.75],
        ),
        pytest.param(
            CosineAnnealingWarmRestartsScheduler(t_0='0.003dur', t_mult=1.5),
            1.0,
            ['0ba', '1000ba', '3000ba', '4500ba', '7500ba', '14250ba'],
            [1.0, 0.75, 1.0, 0.75, 1.0, 1.0],
        ),
        pytest.param(
            CosineAnnealingWarmRestartsScheduler(t_0='30ep', t_mult=2.0, alpha_f=0.5),
            0.5,
            ['0ba', '5000ba', '15000ba', '25000ba'],
            [1.0, 0.875, 1.0, 0.875],
        ),
        pytest.param(
            PolynomialScheduler(power=2.0),
            1.0,
            ['0ba', '100000ba', '200000ba', '500000ba'],
            [1.0, 0.81, 0.64, 0.25],
        ),
        pytest.param(
            PolynomialScheduler(power=2.0, t_max='100ba', alpha_f=0.5),
            1.0,
            ['0ba', '10ba', '20ba', '50ba'],
            [1.0, 0.905, 0.82, 0.625],
        ),
        pytest.param(
            PolynomialScheduler(power=2.0, t_max='100ba', alpha_f=0.5),
            0.5,
            ['0ba', '10ba', '20ba', '50ba'],
            [1.0, 0.82, 0.68, 0.5],
        ),
        pytest.param(
            MultiStepWithWarmupScheduler(t_warmup='10ba', milestones=['20ba', '40ba']),
            1.0,
            ['0ba', '5ba', '15ba', '25ba', '45ba'],
            [0.0, 0.5, 1.0, 0.1, 0.01],
        ),
        pytest.param(
            MultiStepWithWarmupScheduler(t_warmup='10ba', milestones=['2ep', '4ep'], gamma=0.5),
            0.5,
            ['0ba', '5ba', '15ba', '1500ba', '2500ba'],
            [0.0, 0.5, 1.0, 0.5, 0.25],
        ),
        pytest.param(
            MultiStepWithWarmupScheduler(t_warmup='10ba', milestones=['2ep', '4ep'], gamma=0.5, scale_warmup=True),
            0.5,
            ['0ba', '5ba', '15ba', '1500ba', '2500ba'],
            [0.0, 1.0, 1.0, 0.5, 0.25],
        ),
        pytest.param(
            MultiStepWithWarmupScheduler(t_warmup='1000ep', milestones=[]),
            1.0,
            ['0ep', '100ep', '1000ep'],
            [0.0, 0.1, 1.0],
        ),
        pytest.param(
            ConstantWithWarmupScheduler(t_warmup='500ep'),
            1.0,
            ['0ba', '250000ba', '500000ba', '750000ba', '1000000ba'],
            [0.0, 0.5, 1.0, 1.0, 1.0],
        ),
        pytest.param(
            ConstantWithWarmupScheduler(t_warmup='500ep', alpha=3.0),
            1.0,
            ['0ba', '250000ba', '500000ba', '500500ba', '501000ba', '502000ba'],
            [0.0, 1.5, 3.0, 3.0, 3.0, 3.0],
        ),
        pytest.param(
            ConstantWithWarmupScheduler(t_warmup='500ep', alpha=3.0),
            1.0,
            ['0ba', '250000ba', '500000ba', '500500ba', '501000ba', '502000ba'],
            [0.0, 1.5, 3.0, 3.0, 3.0, 3.0],
        ),
        pytest.param(
            ConstantWithWarmupScheduler(t_warmup='0.0005dur'),
            1.0,
            ['0ba', '250ba', '500ba', '499750ba'],
            [0.0, 0.5, 1.0, 1.0],
        ),
        pytest.param(
            ConstantWithWarmupScheduler(t_warmup='500ep', alpha=3.0, t_max='501000ep'),
            0.5,
            ['0ba', '250000ba', '500000ba', '500500ba', '501000ba', '502000ba'],
            [0.0, 1.5, 3.0, 3.0, 3.0, 3.0],
        ),
        pytest.param(ConstantWithWarmupScheduler(t_warmup='1000ep'), 1.0, ['0ep', '100ep', '1000ep'], [0.0, 0.1, 1.0]),
        pytest.param(
            LinearWithWarmupScheduler(t_warmup='500ep'),
            1.0,
            ['0ba', '250000ba', '500000ba', '750000ba', '1000000ba'],
            [0.0, 0.5, 1.0, 0.5, 0.0],
        ),
        pytest.param(
            LinearWithWarmupScheduler(t_warmup='500ep', alpha_i=3.0, alpha_f=2.0, t_max='1002ep'),
            0.5,
            ['0ba', '250000ba', '500000ba', '500500ba', '501000ba', '502000ba'],
            [0.0, 1.5, 3.0, 2.5, 2.0, 2.0],
        ),
        pytest.param(
            LinearWithWarmupScheduler(t_warmup='0.0005dur'),
            1.0,
            ['0ba', '250ba', '500ba', '499750ba'],
            [0.0, 0.5, 1.0, 0.5],
        ),
        pytest.param(
            LinearWithWarmupScheduler(t_warmup='500ba', scale_warmup=False),
            0.5,
            ['0ba', '250ba', '500ba', '249875ba'],
            [0.0, 0.5, 1.0, 0.5],
        ),
        pytest.param(
            LinearWithWarmupScheduler(t_warmup='500ba', scale_warmup=True),
            0.5,
            ['0ba', '125ba', '250ba', '249875ba'],
            [0.0, 0.5, 1.0, 0.5],
        ),
        pytest.param(LinearWithWarmupScheduler(t_warmup='1000ep'), 1.0, ['0ep', '100ep', '1000ep'], [0.0, 0.1, 1.0]),
        pytest.param(
            CosineAnnealingWithWarmupScheduler(t_warmup='0.9dur'),
            1.0,
            ['0ba', '450000ba', '900000ba', '933333ba', '950000ba', '1000000ba'],
            [0.0, 0.5, 1.0, 0.75, 0.5, 0.0],
        ),
        pytest.param(
            CosineAnnealingWithWarmupScheduler(t_warmup='0.9dur', alpha_f=0.5),
            0.01,
            ['0ba', '4500ba', '9000ba', '9333ba', '9500ba', '10000ba'],
            [0.0, 0.5, 1.0, 0.875, 0.75, 0.5],
        ),
        pytest.param(
            CosineAnnealingWithWarmupScheduler(t_warmup='0.9dur', alpha_f=0.5, scale_warmup=True),
            0.01,
            ['0ba', '4500ba', '9000ba', '9333ba', '9500ba', '10000ba'],
            [0.0, 0.5, 1.0, 0.875, 0.75, 0.5],
        ),
        pytest.param(
            CosineAnnealingWithWarmupScheduler(t_warmup='1000ep'),
            1.0,
            ['0ep', '100ep', '1000ep'],
            [0.0, 0.1, 1.0],
        ),
        pytest.param(
            PolynomialWithWarmupScheduler(t_warmup='0.9dur'),
            1.0,
            ['0ba', '450000ba', '900000ba', '913397ba', '929289ba', '1000000ba'],
            [0.0, 0.5, 1.0, 0.75, 0.5, 0.0],
        ),
        pytest.param(
            PolynomialWithWarmupScheduler(t_warmup='0.9dur', alpha_f=0.5),
            0.01,
            ['0ba', '4500ba', '9000ba', '9134ba', '9293ba', '10000ba'],
            [0.0, 0.5, 1.0, 0.875, 0.75, 0.5],
        ),
        pytest.param(
            PolynomialWithWarmupScheduler(t_warmup='0.9dur', alpha_f=0.5, scale_warmup=True),
            0.01,
            ['0ba', '4500ba', '9000ba', '9134ba', '9293ba', '10000ba'],
            [0.0, 0.5, 1.0, 0.875, 0.75, 0.5],
        ),
        pytest.param(
            PolynomialWithWarmupScheduler(t_warmup='1000ep'),
            1.0,
            ['0ep', '100ep', '1000ep'],
            [0.0, 0.1, 1.0],
        ),
    ],
)
def test_scheduler_init(
    scheduler: ComposerScheduler,
    ssr: float,
    test_times: list[str],
    expected_lrs: list[float],
    dummy_schedulers_state: State,
):

    state = dummy_schedulers_state
    assert state.dataloader_len is not None
    assert state.max_duration is not None
    state.max_duration = Time(value=int(state.max_duration.value * ssr), unit=state.max_duration.unit)
    for test_time, expected_lr in zip(test_times, expected_lrs):
        parsed_time = Time.from_timestring(test_time)
        assert parsed_time.unit in [TimeUnit.EPOCH, TimeUnit.BATCH]
        if parsed_time.unit == TimeUnit.EPOCH:
            state.timestamp = state.timestamp.copy(
                epoch=parsed_time,
                batch=Time(int(state.dataloader_len) * int(parsed_time), TimeUnit.BATCH),
            )
        else:
            state.timestamp = state.timestamp.copy(
                batch=parsed_time,
                epoch=Time(int(parsed_time) // int(state.dataloader_len), TimeUnit.EPOCH),
            )

        lr = scheduler(state, ssr)
        assert lr == pytest.approx(expected_lr, abs=1e-3)


@pytest.mark.parametrize(
    'scheduler,ssr,should_raise',
    [
        (StepScheduler(step_size='2ba'), 1.0, None),
        (StepScheduler(step_size='0.2dur', gamma=0.8), 0.5, None),
        (lambda state, ssr=1.0: 0.01 * ssr, 1.5, None),  # lambda's are also allowed as a ComposerScheduler
        (lambda state: 0.01, 1.0, None),  # if the ssr = 1.0, then the lambda need not take the ssr parameter
        (
            lambda state: 0.01,
            1.5,
            ValueError,
        ),  # this should error since the ssr != 1.0 and the lambda doesn't support ssr
    ],
)
def test_scheduler_trains(
    scheduler: ComposerScheduler,
    ssr: float,
    rank_zero_seed: int,
    should_raise: Optional[Type[Exception]],
):
    with pytest.raises(should_raise) if should_raise is not None else contextlib.nullcontext():
        trainer = Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='2ep',
            train_subset_num_batches=5,
            scale_schedule_ratio=ssr,
            schedulers=scheduler,
            seed=rank_zero_seed,
        )
        trainer.fit()


@pytest.mark.parametrize(
    'scheduler_class',
    [
        CosineAnnealingWithWarmupScheduler,
        MultiStepWithWarmupScheduler,
        ConstantWithWarmupScheduler,
        LinearWithWarmupScheduler,
        PolynomialWithWarmupScheduler,
    ],
)
@pytest.mark.parametrize('max_duration_unit', ['tok', 'sp', 'ba', 'ep'])
@pytest.mark.parametrize('warmup_duration_unit', ['ba', 'tok', 'sp', 'ep', 'dur'])
@pytest.mark.parametrize('scheduler_max_unit', ['ba', 'tok', 'sp', 'ep', 'dur'])
@pytest.mark.parametrize('scheduler_max_pct', [0.75, 1.25])
def test_warmup_schedulers_fail_fast(
    scheduler_class: Type[ComposerScheduler],
    max_duration_unit: str,
    warmup_duration_unit: str,
    scheduler_max_unit: str,
    scheduler_max_pct: float,
    dummy_schedulers_state: State,
):
    tokens_per_sample = 8
    samples_per_batch = 16
    batches_per_epoch = 32
    num_epochs = 4
    total_batches = batches_per_epoch * num_epochs
    total_samples = total_batches * samples_per_batch
    total_tokens = total_samples * tokens_per_sample

    warmup_duration_pct = 0.25
    warmup_batches = int(total_batches * warmup_duration_pct)
    warmup_samples = int(total_samples * warmup_duration_pct)
    warmup_tokens = int(total_tokens * warmup_duration_pct)
    warmup_epochs = int(num_epochs * warmup_duration_pct)

    max_batches = int(total_batches * scheduler_max_pct)
    max_samples = int(total_samples * scheduler_max_pct)
    max_tokens = int(total_tokens * scheduler_max_pct)
    max_epochs = int(num_epochs * scheduler_max_pct)

    max_duration_unit_to_str = {
        'tok': f'{total_tokens}tok',
        'sp': f'{total_samples}sp',
        'ba': f'{total_batches}ba',
        'ep': f'{num_epochs}ep',
    }

    warmup_duration_unit_to_str = {
        'tok': f'{warmup_tokens}tok',
        'sp': f'{warmup_samples}sp',
        'ba': f'{warmup_batches}ba',
        'ep': f'{warmup_epochs}ep',
        'dur': f'{warmup_duration_pct}dur',
    }

    scheduler_max_unit_to_str = {
        'tok': f'{max_tokens}tok',
        'sp': f'{max_samples}sp',
        'ba': f'{max_batches}ba',
        'ep': f'{max_epochs}ep',
        'dur': f'{scheduler_max_pct}dur',
    }

    max_duration_str = max_duration_unit_to_str[max_duration_unit]
    warmup_duration_str = warmup_duration_unit_to_str[warmup_duration_unit]
    scheduler_max_str = scheduler_max_unit_to_str[scheduler_max_unit]
    num_steps = total_batches

    if scheduler_class == MultiStepWithWarmupScheduler:
        scheduler = scheduler_class(milestones=['60ba'], t_warmup=warmup_duration_str)  # type: ignore
    else:
        scheduler = scheduler_class(t_warmup=warmup_duration_str, t_max=scheduler_max_str)  # type: ignore

    state = dummy_schedulers_state
    state.max_duration = Time.from_timestring(max_duration_str)
    state.timestamp = Timestamp()
    state.set_dataloader([None] * batches_per_epoch, 'train')

    effective_scheduler_max_unit = scheduler_max_unit if scheduler_max_unit != 'dur' else max_duration_unit
    effective_warmup_duration_unit = warmup_duration_unit if warmup_duration_unit != 'dur' else max_duration_unit
    max_duration_no_epoch = state.max_duration
    if max_duration_unit == 'ep':
        max_duration_no_epoch = Time.from_timestring(max_duration_unit_to_str['ba'])

    error_context = contextlib.nullcontext()
    if (
        hasattr(scheduler, 't_max') and
        not _units_comparable(effective_scheduler_max_unit, effective_warmup_duration_unit)
    ):
        error_context = pytest.raises(ValueError, match='Cannot use warmup scheduler')
    elif (
        hasattr(scheduler, 't_max') and
        _units_comparable(effective_scheduler_max_unit, effective_warmup_duration_unit) and
        _units_comparable(effective_scheduler_max_unit, max_duration_unit) and
        _convert_time(scheduler_max_str, state) < max_duration_no_epoch
    ):
        error_context = pytest.raises(ValueError, match='must be greater than or equal to max_duration')

    with error_context:
        for _ in range(num_steps):
            _ = scheduler(state)
            state.timestamp = state.timestamp.to_next_batch(
                samples=samples_per_batch,
                tokens=tokens_per_sample * samples_per_batch,
            )


def _units_comparable(unit1: str, unit2: str) -> bool:
    if unit1 == unit2:
        return True
    if unit1 == 'ep' and unit2 == 'ba':
        return True
    if unit1 == 'ba' and unit2 == 'ep':
        return True
    return False
