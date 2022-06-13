# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Union

import pytest
import torchmetrics
from torch.utils.data import DataLoader

from composer.core import Event
from composer.core.evaluator import Evaluator, evaluate_periodically
from composer.core.state import State
from composer.core.time import Time, TimeUnit
from composer.datasets.evaluator_hparams import EvaluatorHparams
from composer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from tests.common import EventCounterCallback, RandomClassificationDataset, SimpleModel
from tests.common.datasets import RandomClassificationDatasetHparams


def test_trainer_eval_only():
    # Construct the trainer
    trainer = Trainer(model=SimpleModel())

    # Evaluate the model
    eval_dataloader = DataLoader(dataset=RandomClassificationDataset())
    trainer.eval(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics=torchmetrics.Accuracy(),
    )

    # Assert that there is some accuracy
    assert trainer.state.current_metrics['eval']['Accuracy'] != 0.0


def test_trainer_eval_subset_num_batches():
    # Construct the trainer
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=[event_counter_callback],
    )

    # Evaluate the model
    eval_dataloader = DataLoader(dataset=RandomClassificationDataset())
    trainer.eval(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics=torchmetrics.Accuracy(),
        subset_num_batches=1,
    )

    # Ensure that just one batch was evaluated
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == 1
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == 1


def test_trainer_eval_timestamp():
    # Construct the trainer
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=[event_counter_callback],
    )

    # Evaluate the model
    eval_dataloader = DataLoader(dataset=RandomClassificationDataset())
    trainer.eval(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics=torchmetrics.Accuracy(),
    )

    # Ensure that the eval timestamp matches the number of evaluation events
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == trainer.state.eval_timestamp.batch
    assert trainer.state.eval_timestamp.batch == trainer.state.eval_timestamp.batch_in_epoch

    # Ensure that if we eval again, the eval timestamp was reset

    # Reset the event counter callback
    event_counter_callback.event_to_num_calls = {k: 0 for k in event_counter_callback.event_to_num_calls}

    # Eval again
    trainer.eval(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics=torchmetrics.Accuracy(),
    )
    # Validate the same invariants
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == trainer.state.eval_timestamp.batch
    assert trainer.state.eval_timestamp.batch == trainer.state.eval_timestamp.batch_in_epoch


@pytest.mark.parametrize('eval_at_fit_end', [
    True,
    False,
])
def test_eval_at_fit_end(eval_at_fit_end: bool):
    """Test the `eval_subset_num_batches` and `eval_interval` works when specified on init."""

    # Construct the trainer
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    event_counter_callback = EventCounterCallback()
    eval_interval = '2ep'
    evaluator = Evaluator(
        label='eval',
        dataloader=DataLoader(dataset=RandomClassificationDataset()),
        metrics=torchmetrics.Accuracy(),
    )

    evaluator.eval_interval = evaluate_periodically(eval_interval=eval_interval, eval_at_fit_end=eval_at_fit_end)

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,
        eval_dataloader=evaluator,
        eval_subset_num_batches=1,
        max_duration='3ep',
        callbacks=[event_counter_callback],
    )

    # Train (should evaluate once)
    trainer.fit()

    expected_eval_start_calls = 1
    expected_eval_batch_start_calls = 1

    # depending on eval_at_fit_end, ensure the appropriate amount of calls are invoked
    if eval_at_fit_end:
        # we should have one extra call from eval_at_fit_end
        assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_eval_start_calls + 1
        assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_eval_batch_start_calls + 1
    else:
        assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_eval_start_calls
        assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_eval_batch_start_calls


@pytest.mark.parametrize('eval_dataloader', [
    DataLoader(dataset=RandomClassificationDataset()),
    Evaluator(
        label='eval',
        dataloader=DataLoader(dataset=RandomClassificationDataset()),
        metrics=torchmetrics.Accuracy(),
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
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
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


def test_eval_hparams(composer_trainer_hparams: TrainerHparams):
    """Test that `eval_interval` and `eval_subset_num_batches` work when specified via hparams."""
    # Create the trainer from hparams
    composer_trainer_hparams.eval_interval = '2ep'
    composer_trainer_hparams.eval_subset_num_batches = 2
    composer_trainer_hparams.evaluators = [
        EvaluatorHparams(
            label='eval1',
            eval_interval='3ep',  # will run, since eval_at_fit_end = True
            subset_num_batches=1,
            eval_dataset=RandomClassificationDatasetHparams(),
        ),
        EvaluatorHparams(
            label='eval2',
            eval_dataset=RandomClassificationDatasetHparams(),
            metric_names=['Accuracy'],
        ),
    ]
    composer_trainer_hparams.val_dataset = None
    composer_trainer_hparams.callbacks = [EventCounterCallback()]
    composer_trainer_hparams.max_duration = '2ep'
    trainer = composer_trainer_hparams.initialize_object()

    # Validate that `subset_num_batches` was set correctly
    assert trainer.state.evaluators[0].subset_num_batches == composer_trainer_hparams.evaluators[0].subset_num_batches
    assert trainer.state.evaluators[1].subset_num_batches == composer_trainer_hparams.eval_subset_num_batches

    # Train the model
    trainer.fit()

    # Validate that `eval_interval` and `subset_num_batches` was set correctly for the evaluator that actually
    # ran
    assert 'eval1' in trainer.state.current_metrics
    assert 'eval2' in trainer.state.current_metrics
    event_counter_callback = None
    for callback in trainer.state.callbacks:
        if isinstance(callback, EventCounterCallback):
            event_counter_callback = callback
            break
    assert event_counter_callback is not None
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == 2
    # increment by one for the extra call to `Event.EVAL_BATCH_START` during the evaluation at FIT end.
    assert event_counter_callback.event_to_num_calls[
        Event.EVAL_BATCH_START] == composer_trainer_hparams.eval_subset_num_batches + 1


def test_eval_params_evaluator():
    """Test the `eval_subset_num_batches` and `eval_interval` works when specified as part of an evaluator."""
    # Construct the trainer
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    eval_interval_batches = 1
    eval_subset_num_batches = 2
    eval_dataloader = Evaluator(
        label='eval',
        dataloader=DataLoader(dataset=RandomClassificationDataset()),
        metrics=torchmetrics.Accuracy(),
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
