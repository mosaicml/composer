# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.core import Event
from composer.trainer.trainer_hparams import EvaluatorHparams, TrainerHparams
from tests.common import EventCounterCallback
from tests.hparams.common import DataLoaderHparams, RandomClassificationDatasetHparams, SimpleModelHparams


def test_eval_hparams():
    """Test that `eval_interval` and `eval_subset_num_batches` work when specified via hparams."""
    # Create the trainer from hparams
    composer_trainer_hparams = TrainerHparams(
        model=SimpleModelHparams(),
        train_dataset=RandomClassificationDatasetHparams(),
        dataloader=DataLoaderHparams(
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        ),
        max_duration='2ep',
        eval_batch_size=1,
        train_batch_size=1,
        eval_interval='2ep',
        eval_subset_num_batches=2,
        callbacks=[EventCounterCallback()],
    )

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
    trainer = composer_trainer_hparams.initialize_object()

    # Validate that `subset_num_batches` was set correctly
    assert trainer.state.evaluators[0].subset_num_batches == composer_trainer_hparams.evaluators[0].subset_num_batches
    assert trainer.state.evaluators[1].subset_num_batches == composer_trainer_hparams.eval_subset_num_batches

    # Train the model
    trainer.fit()

    # Validate that `eval_interval` and `subset_num_batches` was set correctly for the evaluator that actually
    # ran
    assert 'eval1' in trainer.state.eval_metrics
    assert 'eval2' in trainer.state.eval_metrics
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
