from typing import Callable, Union

import pytest
import torchmetrics
from torch.utils.data import DataLoader

from composer.core import Event
from composer.core.evaluator import Evaluator
from composer.core.state import State
from composer.core.time import Time, TimeUnit
from composer.datasets.evaluator import EvaluatorHparams
from composer.trainer import Trainer, TrainerHparams
from tests.common import EventCounterCallback, RandomClassificationDataset, SimpleModel
from tests.common.datasets import RandomClassificationDatasetHparams
from tests.common.events import EventCounterCallbackHparams


@pytest.mark.filterwarnings(r"ignore:.*NoSchedulerWarning.*")
@pytest.mark.filterwarnings(r"ignore:.*No `eval_dataloader` was specified.*")
def test_trainer_eval_only():
    # Construct the trainer
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,  # TODO(ravi): Remove in #948
        max_duration='1ep',  # TODO(ravi): Remove in #948
    )

    # Evaluate the model
    eval_dataloader = DataLoader(dataset=RandomClassificationDataset())
    trainer.eval(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics=torchmetrics.Accuracy(),
    )

    # Assert that there is some accuracy
    assert trainer.state.current_metrics['eval']['Accuracy'] != 0.0


@pytest.mark.filterwarnings(r"ignore:.*NoSchedulerWarning.*")
@pytest.mark.filterwarnings(r"ignore:.*No `eval_dataloader` was specified.*")
def test_trainer_eval_subset_num_batches():
    # Construct the trainer
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,  # TODO(ravi): Remove in #948
        max_duration='1ep',  # TODO(ravi): Remove in #948
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


@pytest.mark.parametrize("eval_dataloader", [
    DataLoader(dataset=RandomClassificationDataset()),
    Evaluator(
        label="eval",
        dataloader=DataLoader(dataset=RandomClassificationDataset()),
        metrics=torchmetrics.Accuracy(),
    ),
])
@pytest.mark.parametrize(
    "eval_interval",
    [  # multiple ways of specifying to evaluate once every epoch
        1,
        "1ep",
        Time(1, TimeUnit.EPOCH),
        lambda state, event: event == Event.EPOCH_END,
    ])
@pytest.mark.filterwarnings(r"ignore:.*NoSchedulerWarning.*")
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
    composer_trainer_hparams.eval_interval = "2ep"
    composer_trainer_hparams.eval_subset_num_batches = 2
    composer_trainer_hparams.evaluators = [
        EvaluatorHparams(
            label="eval1",
            eval_interval='3ep',  # should NEVER run, since we train just 2 epochs
            subset_num_batches=1,
            eval_dataset=RandomClassificationDatasetHparams(),
        ),
        EvaluatorHparams(
            label="eval2",
            eval_dataset=RandomClassificationDatasetHparams(),
            metric_names=['Accuracy'],
        ),
    ]
    composer_trainer_hparams.val_dataset = None
    composer_trainer_hparams.callbacks = [EventCounterCallbackHparams()]
    composer_trainer_hparams.max_duration = "2ep"
    trainer = composer_trainer_hparams.initialize_object()

    # Validate that `subset_num_batches` was set correctly
    assert trainer.evaluators[0].subset_num_batches == composer_trainer_hparams.evaluators[0].subset_num_batches
    assert trainer.evaluators[1].subset_num_batches == composer_trainer_hparams.eval_subset_num_batches

    # Train the model
    trainer.fit()

    # Validate that `eval_interval` and `subset_num_batches` was set correctly for the evaluator that actually
    # ran
    assert "eval1" not in trainer.state.current_metrics
    assert "eval2" in trainer.state.current_metrics
    event_counter_callback = None
    for callback in trainer.state.callbacks:
        if isinstance(callback, EventCounterCallback):
            event_counter_callback = callback
            break
    assert event_counter_callback is not None
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == 1
    assert event_counter_callback.event_to_num_calls[
        Event.EVAL_BATCH_START] == composer_trainer_hparams.eval_subset_num_batches


@pytest.mark.filterwarnings(r"ignore:.*NoSchedulerWarning.*")
def test_eval_params_evaluator():
    """Test the `eval_subset_num_batches` and `eval_interval` works when specified as part of an evaluator."""
    # Construct the trainer
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    eval_interval_batches = 1
    eval_subset_num_batches = 2
    eval_dataloader = Evaluator(
        label="eval",
        dataloader=DataLoader(dataset=RandomClassificationDataset()),
        metrics=torchmetrics.Accuracy(),
        eval_interval=f"{eval_interval_batches}ba",
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
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == trainer.state.timer.batch
    assert event_counter_callback.event_to_num_calls[
        Event.EVAL_BATCH_START] == eval_subset_num_batches * trainer.state.timer.batch
