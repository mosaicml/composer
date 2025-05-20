# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.core import DataSpec
from composer.trainer import Trainer
from composer.utils import dist, get_device
from tests.common import device
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset
from tests.common.models import SimpleTransformerClassifier, SimpleTransformerMaskedLM


def test_simple_nlp_classification():
    vocab_size = 100
    sequence_length = 32
    num_classes = 2
    size = 96
    batch_size = 8

    train_dataset = RandomTextClassificationDataset(
        size=size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )
    eval_dataset = RandomTextClassificationDataset(
        size=size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )
    predict_dataset = RandomTextClassificationDataset(
        size=size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )

    model = SimpleTransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, sampler=dist.get_sampler(eval_dataset))
    predict_dataloader = DataLoader(predict_dataset, batch_size=8)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='2ep',
        eval_dataloader=eval_dataloader,
    )

    trainer.fit()
    trainer.eval()

    # Check that there is some train/eval accuracy
    assert trainer.state.train_metrics is not None
    assert trainer.state.train_metrics['MulticlassAccuracy'].compute() != 0.0
    assert trainer.state.eval_metrics['eval']['MulticlassAccuracy'].compute() != 0.0

    predictions = trainer.predict(predict_dataloader)

    # Check that the output predictions are the expected shape
    num_predict_batches_expected = ((size - 1) // batch_size) + 1
    assert len(predictions) == num_predict_batches_expected
    assert predictions[0].shape == (batch_size, 2)


def test_simple_nlp_mlm(tiny_bert_tokenizer, tiny_bert_model):
    transformers = pytest.importorskip('transformers')

    vocab_size = tiny_bert_tokenizer.vocab_size
    sequence_length = 32
    size = 96
    batch_size = 8

    train_dataset = RandomTextLMDataset(
        size=size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        use_keys=True,
    )
    eval_dataset = RandomTextLMDataset(size=size, vocab_size=vocab_size, sequence_length=sequence_length, use_keys=True)
    predict_dataset = RandomTextLMDataset(
        size=size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        use_keys=True,
    )
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)

    model = SimpleTransformerMaskedLM(vocab_size=vocab_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(train_dataset),
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(eval_dataset),
        collate_fn=collator,
    )
    predict_dataloader = DataLoader(predict_dataset, batch_size=8)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='2ep',
        eval_dataloader=eval_dataloader,
    )

    trainer.fit()
    trainer.eval()

    # Check that there is some train/eval cross entropy
    assert trainer.state.train_metrics is not None
    assert trainer.state.train_metrics['LanguageCrossEntropy'].compute() != 0.0
    assert trainer.state.eval_metrics['eval']['LanguageCrossEntropy'].compute() != 0.0

    predictions = trainer.predict(predict_dataloader)

    # Check that the output predictions are the expected shape
    num_predict_batches_expected = ((size - 1) // batch_size) + 1
    assert len(predictions) == num_predict_batches_expected
    assert predictions[0].shape == (batch_size, sequence_length, vocab_size)


@device('gpu')
def test_simple_nlp_mlm_token_batch(tiny_bert_tokenizer, device):
    transformers = pytest.importorskip('transformers')

    vocab_size = tiny_bert_tokenizer.vocab_size
    sequence_length = 32
    size = 96
    batch_size = 8
    device = get_device(device)

    train_dataset = RandomTextLMDataset(
        size=size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        use_keys=True,
        pad_token_id=tiny_bert_tokenizer.pad_token_id,
    )
    for i in range(size):  # Proactively load dataset for consistent randomization
        train_dataset[i]
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer)

    # Get the model's state dict before training starts, so we can reproduce results
    model = SimpleTransformerMaskedLM(vocab_size=vocab_size)
    state_dict = model.state_dict()

    # Set up the data spec that can count the non-padding tokens in a batch
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(train_dataset),
        collate_fn=collator,
    )
    data_spec = DataSpec(
        dataloader=train_dataloader,
        get_num_tokens_in_batch=lambda b: (b['input_ids'] != tiny_bert_tokenizer.pad_token_id).sum().item(),
    )

    trainer = Trainer(
        model=model,
        seed=42,
        train_dataloader=data_spec,
        max_duration='2ep',
        device_train_microbatch_size=batch_size // 2,
        accumulate_train_batch_on_tokens=False,
        device=device,
    )
    trainer.fit()

    # Check that there is some train cross entropy
    assert trainer.state.train_metrics is not None
    cross_entropy = trainer.state.train_metrics['LanguageCrossEntropy'].compute()
    assert cross_entropy != 0.0

    # Set up a trainer that accumulates train loss based on token counts, after reloading original state dict
    model.load_state_dict(state_dict)
    token_trainer = Trainer(
        model=model,
        seed=42,
        train_dataloader=data_spec,
        max_duration='2ep',
        device_train_microbatch_size=batch_size // 2,
        accumulate_train_batch_on_tokens=True,
        device=device,
    )
    token_trainer.fit()

    # Check that there is some train cross entropy
    assert token_trainer.state.train_metrics is not None
    token_cross_entropy = token_trainer.state.train_metrics['LanguageCrossEntropy'].compute()
    assert token_cross_entropy != 0.0

    # Require that the train cross entropies are different between the trainers
    assert cross_entropy != token_cross_entropy

    # Make sure we can reproduce the original cross entropy calculation
    model.load_state_dict(state_dict)
    trainer2 = Trainer(
        model=model,
        seed=42,
        train_dataloader=data_spec,
        max_duration='2ep',
        device_train_microbatch_size=batch_size // 2,
        accumulate_train_batch_on_tokens=False,
        device=device,
    )
    trainer2.fit()
    assert trainer2.state.train_metrics is not None
    assert trainer2.state.train_metrics['LanguageCrossEntropy'].compute() == cross_entropy


@device('gpu')
def test_simple_nlp_mlm_loss_gen_token_batch(tiny_bert_tokenizer, device):
    transformers = pytest.importorskip('transformers')

    vocab_size = tiny_bert_tokenizer.vocab_size
    sequence_length = 32
    size = 96
    batch_size = 8
    device = get_device(device)

    train_dataset = RandomTextLMDataset(
        size=size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        use_keys=True,
        pad_token_id=tiny_bert_tokenizer.pad_token_id,
    )
    for i in range(size):  # Proactively load dataset for consistent randomization
        train_dataset[i]
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer)

    # Get the model's state dict before training starts, so we can reproduce results
    model = SimpleTransformerMaskedLM(vocab_size=vocab_size)
    state_dict = model.state_dict()

    # Set up the data spec that can count the non-padding tokens in a batch
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(train_dataset),
        collate_fn=collator,
    )
    data_spec = DataSpec(
        dataloader=train_dataloader,
        get_num_tokens_in_batch=lambda b: (b['input_ids'] != tiny_bert_tokenizer.pad_token_id).sum().item(),
    )

    # Arbitrarily divide num tokens by 2 to simulate loss-generating tokens
    loss_gen_data_spec = DataSpec(
        dataloader=train_dataloader,
        get_num_tokens_in_batch=lambda b: {
            'total': (b['input_ids'] != tiny_bert_tokenizer.pad_token_id).sum().item(),
            'loss_generating': (b['input_ids'] != tiny_bert_tokenizer.pad_token_id).sum().item() // 2,
        },
    )

    trainer = Trainer(
        model=model,
        seed=42,
        train_dataloader=data_spec,
        max_duration='2ep',
        device_train_microbatch_size=batch_size // 2,
        accumulate_train_batch_on_tokens=False,
        device=device,
    )
    trainer.fit()

    # Check that there is some train cross entropy
    assert trainer.state.train_metrics is not None
    cross_entropy = trainer.state.train_metrics['LanguageCrossEntropy'].compute()
    assert cross_entropy != 0.0

    # Set up a trainer that accumulates train loss based on token counts, after reloading original state dict
    model.load_state_dict(state_dict)
    token_trainer = Trainer(
        model=model,
        seed=42,
        train_dataloader=loss_gen_data_spec,
        max_duration='2ep',
        device_train_microbatch_size=batch_size // 2,
        accumulate_train_batch_on_tokens=True,
        device=device,
    )
    token_trainer.fit()

    # Check that there is some train cross entropy
    assert token_trainer.state.train_metrics is not None
    token_cross_entropy = token_trainer.state.train_metrics['LanguageCrossEntropy'].compute()
    assert token_cross_entropy != 0.0

    # Require that the train cross entropies are different between the trainers
    assert cross_entropy != token_cross_entropy
