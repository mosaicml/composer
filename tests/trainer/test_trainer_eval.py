# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Union

import pytest
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from composer.core import Event
from composer.core.evaluator import Evaluator, evaluate_periodically
from composer.core.state import State
from composer.core.time import Time, TimeUnit
from composer.trainer import Trainer
from composer.utils import dist
from tests.common import EventCounterCallback, RandomClassificationDataset, SimpleModel


def test_eval():
    # Construct the trainer
    dataset = RandomClassificationDataset()
    trainer = Trainer(
        eval_dataloader=DataLoader(
            dataset=dataset,
            sampler=dist.get_sampler(dataset),
        ),
        model=SimpleModel(),
    )

    # Evaluate the model
    trainer.eval()

    # Assert that there is some accuracy
    assert trainer.state.eval_metrics['eval']['Accuracy'].compute() != 0.0


def test_eval_call():
    # Construct the trainer
    trainer = Trainer(model=SimpleModel(),)

    # Evaluate the model
    dataset = RandomClassificationDataset()
    trainer.eval(eval_dataloader=DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
    ))

    # Assert that there is some accuracy
    assert trainer.state.eval_metrics['eval']['Accuracy'].compute() != 0.0


def test_eval_deprecation_error():
    # Construct the trainer
    trainer = Trainer(model=SimpleModel(),)

    with pytest.raises(ValueError):
        dataset = RandomClassificationDataset()
        trainer.eval(
            dataloader=DataLoader(
                dataset=dataset,
                sampler=dist.get_sampler(dataset),
            ),
            dataloader_label='test',
            metrics=Accuracy(),
        )


def test_eval_type_error():
    # Construct the trainer
    trainer = Trainer(model=SimpleModel(),)

    with pytest.raises(TypeError):
        trainer.eval(unknown_kwarg=None,)


def test_trainer_eval_loop():
    # Construct the trainer
    trainer = Trainer(model=SimpleModel())

    # Evaluate the model
    dataset = RandomClassificationDataset()
    eval_dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
    )
    trainer._eval_loop(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics={'Accuracy': Accuracy()},
    )

    # Assert that there is some accuracy
    assert trainer.state.eval_metrics['eval']['Accuracy'].compute() != 0.0


def test_trainer_eval_subset_num_batches():
    # Construct the trainer
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=[event_counter_callback],
    )

    # Evaluate the model
    dataset = RandomClassificationDataset()
    eval_dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
    )
    trainer.eval(
        eval_dataloader=eval_dataloader,
        subset_num_batches=1,
    )

    # Ensure that just one batch was evaluated
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == 1
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == 1


@pytest.mark.filterwarnings(r'ignore:eval_dataloader label:UserWarning')
def test_trainer_eval_timestamp():
    # Construct the trainer
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=[event_counter_callback],
    )

    # Evaluate the model
    dataset = RandomClassificationDataset()
    eval_dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
    )
    trainer.eval(eval_dataloader=eval_dataloader)

    # Ensure that the eval timestamp matches the number of evaluation events
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == trainer.state.eval_timestamp.batch
    assert trainer.state.eval_timestamp.batch == trainer.state.eval_timestamp.batch_in_epoch

    # Ensure that if we eval again, the eval timestamp was reset

    # Reset the event counter callback
    event_counter_callback.event_to_num_calls = {k: 0 for k in event_counter_callback.event_to_num_calls}

    # Eval again
    trainer.eval(eval_dataloader=eval_dataloader)
    # Validate the same invariants
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == trainer.state.eval_timestamp.batch
    assert trainer.state.eval_timestamp.batch == trainer.state.eval_timestamp.batch_in_epoch


@pytest.mark.parametrize(('eval_interval', 'max_duration', 'eval_at_fit_end', 'expected_eval_start_calls',
                          'expected_eval_batch_start_calls'), [
                              (1, '5ep', True, 4, 4),
                              (Time(2, TimeUnit.EPOCH), '8ep', False, 4, 4),
                              (Time(100, TimeUnit.BATCH), '8ep', False, 4, 4),
                              (Time(0.25, TimeUnit.DURATION), '4ep', False, 4, 4),
                              ('1ep', '4ep', True, 3, 3),
                              ('50ba', '4ep', False, 4, 4),
                              ('50ba', '100ba', False, 2, 2),
                              ('0.35dur', '4ep', True, 2, 2),
                              ('0.01dur', '1000ba', False, 100, 100),
                              ('0.10dur', '700sp', True, 9, 9),
                              ('0.05dur', '700sp', False, 20, 20),
                          ])
def test_eval_at_fit_end(eval_interval: Union[str, Time, int], max_duration: str, eval_at_fit_end: bool,
                         expected_eval_start_calls: int, expected_eval_batch_start_calls: int):
    """Test the `eval_subset_num_batches` and `eval_interval` works when specified on init."""

    # Construct the trainer
    train_dataset = RandomClassificationDataset()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        sampler=dist.get_sampler(train_dataset),
    )
    event_counter_callback = EventCounterCallback()
    eval_interval = eval_interval
    eval_dataset = RandomClassificationDataset()
    evaluator = Evaluator(
        label='eval',
        dataloader=DataLoader(
            dataset=eval_dataset,
            sampler=dist.get_sampler(eval_dataset),
        ),
        metric_names=['Accuracy'],
    )

    evaluator.eval_interval = evaluate_periodically(
        eval_interval=eval_interval,
        eval_at_fit_end=eval_at_fit_end,
    )

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,
        eval_dataloader=evaluator,
        eval_subset_num_batches=1,
        max_duration=max_duration,
        callbacks=[event_counter_callback],
    )

    # Train (should evaluate once)
    trainer.fit()

    # depending on eval_at_fit_end, ensure the appropriate amount of calls are invoked
    if eval_at_fit_end:
        # we should have one extra call from eval_at_fit_end
        assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_eval_start_calls + 1
        assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_eval_batch_start_calls + 1
    else:
        assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_eval_start_calls
        assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_eval_batch_start_calls


def _get_classification_dataloader():
    dataset = RandomClassificationDataset()
    return DataLoader(dataset, sampler=dist.get_sampler(dataset))


@pytest.mark.parametrize('eval_dataloader', [
    _get_classification_dataloader(),
    Evaluator(
        label='eval',
        dataloader=_get_classification_dataloader(),
        metric_names=['Accuracy'],
    ),
])
@pytest.mark.parametrize(
    'eval_interval',
    [  # multiple ways of specifying to evaluate once every epoch
        1,
        '1ep',
        Time(1, TimeUnit.EPOCH),
        lambda state, event: event == Event.EPOCH_END,
    ])
def test_eval_params_init(
    eval_dataloader: Union[DataLoader, Evaluator],
    eval_interval: Union[Time, str, int, Callable[[State, Event], bool]],
):
    """Test the `eval_subset_num_batches` and `eval_interval` works when specified on init."""

    # Construct the trainer
    train_dataset = RandomClassificationDataset()
    train_dataloader = DataLoader(train_dataset, sampler=dist.get_sampler(train_dataset))
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_subset_num_batches=1,
        max_duration='1ep',
        callbacks=[event_counter_callback],
        eval_interval=eval_interval,
    )

    # Train (should evaluate once)
    trainer.fit()

    # Assert that the evaluator was indeed called only once
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == 1
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == 1


def test_eval_params_evaluator():
    """Test the `eval_subset_num_batches` and `eval_interval` works when specified as part of an evaluator."""
    # Construct the trainer
    train_dataset = RandomClassificationDataset()
    train_dataloader = DataLoader(train_dataset, sampler=dist.get_sampler(train_dataset))
    eval_interval_batches = 1
    eval_subset_num_batches = 2
    eval_dataset = RandomClassificationDataset()
    eval_dataloader = Evaluator(
        label='eval',
        dataloader=DataLoader(
            dataset=eval_dataset,
            sampler=dist.get_sampler(eval_dataset),
        ),
        metric_names=['Accuracy'],
        eval_interval=f'{eval_interval_batches}ba',
        subset_num_batches=eval_subset_num_batches,
    )
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration='1ep',
        callbacks=[event_counter_callback],
        # These parameters should be ignored since `subset_num_batches` is specified as part of the Evaluator
        eval_subset_num_batches=1,
        eval_interval='1ep',
    )

    # Train the model (should evaluate once every batch)
    trainer.fit()

    # Assert that the evaluator ran once every batch
    # (and not the `eval_interval` as specified for the Trainer)
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == trainer.state.timestamp.batch
    assert event_counter_callback.event_to_num_calls[
        Event.EVAL_BATCH_START] == eval_subset_num_batches * trainer.state.timestamp.batch
