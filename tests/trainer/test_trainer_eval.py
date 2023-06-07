# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
import pathlib
from typing import Callable, Optional, Union

import pytest
from torch.utils.data import DataLoader

from composer.core import Algorithm, Event
from composer.core.evaluator import Evaluator, evaluate_periodically
from composer.core.state import State
from composer.core.time import Time, TimeUnit
from composer.trainer import Trainer
from composer.utils import dist
from tests.common import (EventCounterCallback, ParityDataset, RandomClassificationDataset, RandomTextLMDataset,
                          SimpleModel, SimpleTransformerMaskedLM, ZeroModel, world_size)


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
    assert trainer.state.eval_metrics['eval']['MulticlassAccuracy'].compute() != 0.0


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
    assert trainer.state.eval_metrics['eval']['MulticlassAccuracy'].compute() != 0.0


@world_size(1, 2)
@pytest.mark.parametrize('size', [12, 13, 14, 15, 16])
@pytest.mark.parametrize('batch_size', [1, 2, 3, 4, 6])
@pytest.mark.filterwarnings(r'ignore:Cannot split tensor of length.*:UserWarning')
def test_eval_with_nondivisible_dataset(world_size: int, size: int, batch_size: int):
    # Construct the trainer
    trainer = Trainer(model=ZeroModel())

    # Evaluate the model
    dataset = ParityDataset(size=size)
    trainer.eval(eval_dataloader=DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(dataset),
    ))

    expected_acc = 1 - (size // 2) / size
    metric = trainer.state.eval_metrics['eval']['MulticlassAccuracy']
    assert metric.compute() - expected_acc < 1e-5
    count = metric.tp + metric.fn  # type: ignore
    dist.all_reduce(count)
    assert count.item() == size


def test_eval_call_with_trainer_evaluators():
    trainer_dataset = RandomClassificationDataset()
    trainer_evaluator = Evaluator(label='trainer',
                                  dataloader=DataLoader(
                                      dataset=trainer_dataset,
                                      sampler=dist.get_sampler(trainer_dataset),
                                  ))

    eval_call_dataset = RandomClassificationDataset()
    eval_call_evaluator = Evaluator(label='eval_call',
                                    dataloader=DataLoader(dataset=eval_call_dataset,
                                                          sampler=dist.get_sampler(eval_call_dataset)))
    # Construct the trainer
    trainer = Trainer(model=SimpleModel(), eval_dataloader=trainer_evaluator)

    # Empty eval call.
    trainer.eval()

    # Check trainer_evaluator is not deleted.
    assert trainer_evaluator in trainer.state.evaluators

    # Eval call with an evaluator passed.
    trainer.eval(eval_dataloader=eval_call_evaluator)

    # Evaluators passed to constructor permanently reside in trainer.state.evaluators.
    # Check trainer_evaluator is NOT deleted.
    assert trainer_evaluator in trainer.state.evaluators
    # Evaluators passed to eval temporarily reside in trainer.state.evaluators for the duration
    # of evaluation.
    # Check eval_call_evaluator IS deleted.
    assert eval_call_evaluator not in trainer.state.evaluators


@pytest.mark.parametrize('evaluator_on_init,subset_on_init', [[True, True], [True, False], [False, False]])
def test_trainer_eval_subset_num_batches(evaluator_on_init: bool, subset_on_init: bool):
    dataset = RandomClassificationDataset()
    eval_dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
    )

    # Construct the trainer
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=[event_counter_callback],
        eval_dataloader=eval_dataloader if evaluator_on_init else None,
        eval_subset_num_batches=1 if subset_on_init else -1,
    )

    # Evaluate the model
    trainer.eval(
        eval_dataloader=eval_dataloader if not evaluator_on_init else None,
        subset_num_batches=1 if not subset_on_init else -1,
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


@pytest.mark.parametrize('eval_interval', ['1tok', '64tok', '65tok'])
@pytest.mark.parametrize('batch_size', [1, 4, 5])
@pytest.mark.parametrize('sequence_length', [1, 16])
def test_eval_token_interval(tiny_bert_tokenizer, eval_interval: str, batch_size: int, sequence_length: int,
                             tmp_path: pathlib.Path):
    """Tests that the trainer evaluates the model at the correct intervals when using token-based intervals."""
    tokens_per_batch = batch_size * sequence_length
    max_duration_time = Time.from_timestring('5ba')
    eval_interval_time = Time.from_timestring(eval_interval)
    max_duration_tokens = max_duration_time.value * tokens_per_batch

    # calculate the expected number of evals
    last_token_iter = 0
    next_multiple = eval_interval_time.value
    expected_num_evals = 0
    last_multiple_added = -1
    for token_iter in range(0, max_duration_tokens + tokens_per_batch, tokens_per_batch):
        if last_token_iter < next_multiple <= token_iter:
            last_multiple_added = next_multiple
            expected_num_evals += 1
        last_token_iter = token_iter
        while next_multiple <= last_token_iter:
            next_multiple += eval_interval_time.value

    if last_multiple_added + tokens_per_batch <= max_duration_tokens:
        expected_num_evals += 1

    num_eval_batches = 2
    expected_batch_evals = expected_num_evals * num_eval_batches

    transformers = pytest.importorskip('transformers')
    model = SimpleTransformerMaskedLM(vocab_size=tiny_bert_tokenizer.vocab_size)
    pretraining_train_dataset = RandomTextLMDataset(size=100,
                                                    vocab_size=tiny_bert_tokenizer.vocab_size,
                                                    sequence_length=sequence_length,
                                                    use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
    dataloader = DataLoader(pretraining_train_dataset,
                            batch_size=batch_size,
                            sampler=dist.get_sampler(pretraining_train_dataset),
                            collate_fn=collator)
    eval_dataloader = DataLoader(pretraining_train_dataset,
                                 batch_size=batch_size,
                                 sampler=dist.get_sampler(pretraining_train_dataset),
                                 collate_fn=collator)

    event_counter_callback = EventCounterCallback()
    trainer = Trainer(model=model,
                      train_dataloader=dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration=max_duration_time,
                      eval_interval=eval_interval_time,
                      callbacks=[event_counter_callback],
                      eval_subset_num_batches=num_eval_batches)
    trainer.fit()

    # we should have one extra call from eval_at_fit_end
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_num_evals
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_batch_evals


@pytest.mark.parametrize('eval_interval', ['1sp', '4sp', '5sp'])
@pytest.mark.parametrize('batch_size', [1, 4, 5])
@pytest.mark.parametrize('sequence_length', [1, 16])
def test_eval_sample_interval(tiny_bert_tokenizer, eval_interval: str, batch_size: int, sequence_length: int,
                              tmp_path: pathlib.Path):
    """Tests that the trainer evaluates the model at the correct intervals when using sample-based intervals."""
    max_duration_time = Time.from_timestring('5ba')
    eval_interval_time = Time.from_timestring(eval_interval)
    max_duration_samples = max_duration_time.value * batch_size

    # calculate the expected number of evals
    last_sample_iter = 0
    next_multiple = eval_interval_time.value
    expected_num_evals = 0
    last_multiple_added = -1
    for sample_iter in range(0, max_duration_samples + batch_size, batch_size):
        if last_sample_iter < next_multiple <= sample_iter:
            last_multiple_added = next_multiple
            expected_num_evals += 1
        last_token_iter = sample_iter
        while next_multiple <= last_token_iter:
            next_multiple += eval_interval_time.value

    if last_multiple_added + batch_size <= max_duration_samples:
        expected_num_evals += 1

    num_eval_batches = 2
    expected_batch_evals = expected_num_evals * num_eval_batches

    transformers = pytest.importorskip('transformers')
    model = SimpleTransformerMaskedLM(vocab_size=tiny_bert_tokenizer.vocab_size)
    pretraining_train_dataset = RandomTextLMDataset(size=100,
                                                    vocab_size=tiny_bert_tokenizer.vocab_size,
                                                    sequence_length=sequence_length,
                                                    use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
    dataloader = DataLoader(pretraining_train_dataset,
                            batch_size=batch_size,
                            sampler=dist.get_sampler(pretraining_train_dataset),
                            collate_fn=collator)
    eval_dataloader = DataLoader(pretraining_train_dataset,
                                 batch_size=batch_size,
                                 sampler=dist.get_sampler(pretraining_train_dataset),
                                 collate_fn=collator)

    event_counter_callback = EventCounterCallback()
    trainer = Trainer(model=model,
                      train_dataloader=dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration=max_duration_time,
                      eval_interval=eval_interval_time,
                      callbacks=[event_counter_callback],
                      eval_subset_num_batches=num_eval_batches)
    trainer.fit()

    # we should have one extra call from eval_at_fit_end
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_num_evals
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_batch_evals


@pytest.mark.parametrize('max_duration', ['1tok', '213tok', '512tok', '513tok'])
@pytest.mark.parametrize('eval_interval', ['0.1dur', '0.57dur'])
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('sequence_length', [1, 16])
def test_eval_dur_interval_token_max(tiny_bert_tokenizer, eval_interval: str, max_duration: str, batch_size: int,
                                     sequence_length: int):
    """Tests that the trainer evaluates the model at the correct intervals when using duration-based intervals, with max_duration in tokens."""
    max_duration_time = Time.from_timestring(max_duration)
    eval_interval_time = Time.from_timestring(eval_interval)
    tokens_per_batch = batch_size * sequence_length
    eval_interval_tokens = math.ceil(max_duration_time.value * eval_interval_time.value)

    # calculate the expected number of evals
    last_token_iter = 0
    next_multiple = eval_interval_tokens
    expected_num_evals = 0
    last_multiple_added = -1
    for token_iter in range(0, max_duration_time.value + tokens_per_batch, tokens_per_batch):
        if last_token_iter < next_multiple <= token_iter:
            last_multiple_added = next_multiple
            expected_num_evals += 1
        last_token_iter = token_iter
        while next_multiple <= last_token_iter:
            next_multiple += eval_interval_tokens

    if last_multiple_added + tokens_per_batch <= max_duration_time.value:
        expected_num_evals += 1

    num_eval_batches = 2
    expected_batch_evals = expected_num_evals * num_eval_batches

    transformers = pytest.importorskip('transformers')
    model = SimpleTransformerMaskedLM(vocab_size=tiny_bert_tokenizer.vocab_size)
    pretraining_train_dataset = RandomTextLMDataset(size=100,
                                                    vocab_size=tiny_bert_tokenizer.vocab_size,
                                                    sequence_length=sequence_length,
                                                    use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
    dataloader = DataLoader(pretraining_train_dataset,
                            batch_size=batch_size,
                            sampler=dist.get_sampler(pretraining_train_dataset),
                            collate_fn=collator)
    eval_dataloader = DataLoader(pretraining_train_dataset,
                                 batch_size=batch_size,
                                 sampler=dist.get_sampler(pretraining_train_dataset),
                                 collate_fn=collator)

    event_counter_callback = EventCounterCallback()
    trainer = Trainer(model=model,
                      train_dataloader=dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration=max_duration_time,
                      eval_interval=eval_interval_time,
                      callbacks=[event_counter_callback],
                      eval_subset_num_batches=num_eval_batches)
    trainer.fit()

    # we should have one extra call from eval_at_fit_end
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_num_evals
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_batch_evals


@pytest.mark.parametrize('max_duration', ['1sp', '13sp', '32sp', '33sp'])
@pytest.mark.parametrize('eval_interval', ['0.1dur', '0.57dur'])
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('sequence_length', [1, 16])
def test_eval_dur_interval_sample_max(tiny_bert_tokenizer, eval_interval: str, max_duration: str, batch_size: int,
                                      sequence_length: int):
    """Tests that the trainer evaluates the model at the correct intervals when using duration-based intervals, with max_duration in tokens."""
    max_duration_time = Time.from_timestring(max_duration)
    eval_interval_time = Time.from_timestring(eval_interval)
    samples_per_batch = batch_size
    eval_interval_samples = math.ceil(max_duration_time.value * eval_interval_time.value)

    # calculate the expected number of evals
    last_sample_iter = 0
    next_multiple = eval_interval_samples
    expected_num_evals = 0
    last_multiple_added = -1
    for sample_iter in range(0, max_duration_time.value + samples_per_batch, samples_per_batch):
        if last_sample_iter < next_multiple <= sample_iter:
            last_multiple_added = next_multiple
            expected_num_evals += 1
        last_sample_iter = sample_iter
        while next_multiple <= last_sample_iter:
            next_multiple += eval_interval_samples

    if last_multiple_added + samples_per_batch <= max_duration_time.value:
        expected_num_evals += 1

    num_eval_batches = 2
    expected_batch_evals = expected_num_evals * num_eval_batches

    transformers = pytest.importorskip('transformers')
    model = SimpleTransformerMaskedLM(vocab_size=tiny_bert_tokenizer.vocab_size)
    pretraining_train_dataset = RandomTextLMDataset(size=100,
                                                    vocab_size=tiny_bert_tokenizer.vocab_size,
                                                    sequence_length=sequence_length,
                                                    use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
    dataloader = DataLoader(pretraining_train_dataset,
                            batch_size=batch_size,
                            sampler=dist.get_sampler(pretraining_train_dataset),
                            collate_fn=collator)
    eval_dataloader = DataLoader(pretraining_train_dataset,
                                 batch_size=batch_size,
                                 sampler=dist.get_sampler(pretraining_train_dataset),
                                 collate_fn=collator)

    event_counter_callback = EventCounterCallback()
    trainer = Trainer(model=model,
                      train_dataloader=dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration=max_duration_time,
                      eval_interval=eval_interval_time,
                      callbacks=[event_counter_callback],
                      eval_subset_num_batches=num_eval_batches)
    trainer.fit()

    # we should have one extra call from eval_at_fit_end
    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_num_evals
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == expected_batch_evals


@pytest.mark.parametrize(('eval_interval', 'max_duration', 'eval_at_fit_end', 'expected_eval_start_calls',
                          'expected_eval_batch_start_calls'), [
                              (1, '5ep', True, 4, 4),
                              (Time(2, TimeUnit.EPOCH), '8ep', False, 4, 4),
                              (Time(10, TimeUnit.BATCH), '8ep', False, 4, 4),
                              (Time(0.25, TimeUnit.DURATION), '4ep', False, 4, 4),
                              ('1ep', '4ep', True, 3, 3),
                              ('5ba', '4ep', False, 4, 4),
                              ('5ba', '10ba', False, 2, 2),
                              ('0.35dur', '4ep', True, 2, 2),
                              ('0.01dur', '100ba', False, 100, 100),
                              ('0.10dur', '70sp', True, 9, 9),
                              ('0.05dur', '80sp', False, 20, 20),
                          ])
def test_eval_at_fit_end(eval_interval: Union[str, Time, int], max_duration: str, eval_at_fit_end: bool,
                         expected_eval_start_calls: int, expected_eval_batch_start_calls: int):
    """Test the `eval_subset_num_batches` and `eval_interval` works when specified on init."""

    # Construct the trainer
    train_dataset = RandomClassificationDataset(size=10)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        sampler=dist.get_sampler(train_dataset),
    )
    event_counter_callback = EventCounterCallback()
    eval_interval = eval_interval
    eval_dataset = RandomClassificationDataset(size=10)
    evaluator = Evaluator(
        label='eval',
        dataloader=DataLoader(
            dataset=eval_dataset,
            sampler=dist.get_sampler(eval_dataset),
        ),
        metric_names=['MulticlassAccuracy'],
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
        metric_names=['MulticlassAccuracy'],
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
        metric_names=['MulticlassAccuracy'],
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


class InfiniteDataloader(DataLoader):
    """Infinite dataloader that never raises StopIteration."""

    def __iter__(self):
        while True:
            for batch in super().__iter__():
                yield batch

    def __len__(self) -> Optional[int]:
        return None


@pytest.mark.parametrize('eval_subset_num_batches,success', [[None, False], [-1, False], [1, True]])
def test_infinite_eval_dataloader(eval_subset_num_batches, success):
    """Test the `eval_subset_num_batches` is required with infinite dataloader."""
    # Construct the trainer
    train_dataset = RandomClassificationDataset()
    train_dataloader = DataLoader(train_dataset, sampler=dist.get_sampler(train_dataset))
    eval_dataset = RandomClassificationDataset()
    eval_dataloader = InfiniteDataloader(eval_dataset, sampler=dist.get_sampler(eval_dataset))

    with contextlib.nullcontext() if success else pytest.raises(ValueError):
        Trainer(
            model=SimpleModel(),
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_duration='1ep',
            eval_subset_num_batches=eval_subset_num_batches,
        )


class BreakBatchAlgorithm(Algorithm):

    def __init__(self):
        super().__init__()

    def match(self, event, state):
        return event == Event.EVAL_BEFORE_FORWARD

    def apply(self, event, state, logger):
        del event, logger  # unused
        state.batch = None


@pytest.mark.parametrize('add_algorithm', [True, False])
def test_eval_batch_can_be_modified(add_algorithm: bool):
    train_dataset = RandomClassificationDataset(size=8)
    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=dist.get_sampler(train_dataset))
    eval_dataset = RandomClassificationDataset(size=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, sampler=dist.get_sampler(eval_dataset))

    with contextlib.nullcontext() if not add_algorithm else pytest.raises(TypeError):
        trainer = Trainer(model=SimpleModel(),
                          train_dataloader=train_dataloader,
                          eval_dataloader=eval_dataloader,
                          max_duration='1ep',
                          algorithms=[BreakBatchAlgorithm()] if add_algorithm else [])
        trainer.eval()


@pytest.mark.parametrize('metric_names', ['MulticlassAccuracy', ['MulticlassAccuracy']])
def test_evaluator_metric_names_string_errors(metric_names):
    eval_dataset = RandomClassificationDataset(size=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, sampler=dist.get_sampler(eval_dataset))

    context = contextlib.nullcontext() if isinstance(metric_names, list) else pytest.raises(
        ValueError, match='should be a list of strings')
    with context:
        _ = Evaluator(label='evaluator', dataloader=eval_dataloader, metric_names=metric_names)


# Test that trainer throws a value error if the eval is passed a mixed list of evaluators/dataloaders
def test_evaluator_dataloader_value_error():
    eval_dataset = RandomClassificationDataset(size=8)
    eval_data1 = DataLoader(eval_dataset, batch_size=4, sampler=dist.get_sampler(eval_dataset))
    eval_data2 = Evaluator(label='eval', dataloader=eval_data1, metric_names=['MulticlassAccuracy'])

    eval_dataloader = (eval_data1, eval_data2)
    with pytest.raises(ValueError,
                       match='Mixing Evaluator with other classes is not allowed, please wrap'
                       'all other classes with the Evaluator class.') as _:
        _ = Trainer(model=SimpleModel(), train_dataloader=None, eval_dataloader=eval_dataloader, max_duration='1ep')
