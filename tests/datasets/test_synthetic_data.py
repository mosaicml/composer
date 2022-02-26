# Copyright 2021 MosaicML. All Rights Reserved.

import functools
import operator
from typing import Optional

import pytest
from transformers import BertTokenizer, GPT2Tokenizer

from composer.datasets.synthetic import (SyntheticBatchPairDataset, SyntheticDataLabelType, SyntheticDataType,
                                         SyntheticHFDataset, SyntheticPILDataset, generate_synthetic_tokenizer)


@pytest.mark.parametrize('num_samples', [50])
@pytest.mark.parametrize('chars_per_sample', [128])
@pytest.mark.parametrize('column_names', [['sentence'], ['sentence1', 'sentence2']])
@pytest.mark.parametrize('tokenizer_family', ['bert', 'gpt2'])
def test_synthetic_hf_dataset_creation(num_samples: int, chars_per_sample: int, column_names: list,
                                       tokenizer_family: str):
    dataset_generator = SyntheticHFDataset(num_samples=num_samples,
                                           chars_per_sample=chars_per_sample,
                                           column_names=column_names)

    sample = dataset_generator.generate_sample()
    assert len(sample) == chars_per_sample

    dataset = dataset_generator.generate_dataset()
    assert len(dataset) == num_samples
    assert len(dataset[column_names[0]][0]) == chars_per_sample
    assert dataset.column_names == (column_names + ['idx'])

    # build the tokenizer
    tokenizer = generate_synthetic_tokenizer(tokenizer_family, dataset=dataset)
    # verifying the input ids are a part of the tokenizer
    assert 'input_ids' in tokenizer.model_input_names

    # test tokenizing the dataset
    dataset = dataset.map(lambda inp: tokenizer(
        text=inp[column_names[0]], padding="max_length", max_length=chars_per_sample, truncation=True),
                          batched=True,
                          num_proc=1,
                          keep_in_memory=True)

    # verify datapoints are correct
    assert 'input_ids' in dataset.column_names
    x = dataset['input_ids'][0]
    assert len(x) == chars_per_sample

    # add some tokenizer-specific tests
    if tokenizer_family == "bert":
        assert x[0] == tokenizer.cls_token_id
        assert tokenizer.sep_token_id in x

    # since our tokenization max_length==chars_per_sample, we should always have padding tokens due to extra space
    assert x[-1] == tokenizer.pad_token_id


@pytest.mark.parametrize('tokenizer_family', ["bert", "gpt2"])
@pytest.mark.parametrize('vocab_size', [512])
def test_synthetic_tokenizer_creation(tokenizer_family, vocab_size):
    tokenizer = generate_synthetic_tokenizer(tokenizer_family=tokenizer_family, vocab_size=vocab_size)
    if model == "bert":
        assert isinstance(tokenizer, BertTokenizer)
    elif model == "gpt2":
        assert isinstance(tokenizer, GPT2Tokenizer)

    assert tokenizer.vocab_size == vocab_size
    assert tokenizer.pad_token_id == 0
    if tokenizer_family == "bert":
        assert tokenizer.cls_token is not None
        assert tokenizer.sep_token is not None
    elif tokenizer_family == "gpt2":
        assert tokenizer.eos_token is not None


@pytest.mark.parametrize('data_type', [
    SyntheticDataType.GAUSSIAN,
    SyntheticDataType.SEPARABLE,
])
@pytest.mark.parametrize('label_type', [
    SyntheticDataLabelType.CLASSIFICATION_ONE_HOT,
    SyntheticDataLabelType.CLASSIFICATION_INT,
])
def test_synthetic_batch_pair_creation(data_type: SyntheticDataType, label_type: SyntheticDataLabelType):
    if data_type == SyntheticDataType.SEPARABLE:
        if label_type != SyntheticDataLabelType.CLASSIFICATION_INT:
            pytest.skip("Seperable data requires classification int labels")
        num_classes = 2
        label_shape = None
    else:
        num_classes = 10
        label_shape = (1, 10, 12)
        # run run
        return

    dataset_size = 1000
    data_shape = (3, 32, 32)
    num_samples_to_create = 10
    dataset = SyntheticBatchPairDataset(total_dataset_size=dataset_size,
                                        data_shape=data_shape,
                                        num_unique_samples_to_create=num_samples_to_create,
                                        data_type=data_type,
                                        label_type=label_type,
                                        num_classes=num_classes,
                                        label_shape=label_shape)
    assert len(dataset) == dataset_size

    # verify datapoints are correct
    x, y = dataset[0]
    assert x.size() == data_shape
    if label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
        assert isinstance(y.item(), int)
    elif label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
        assert y.size() == (num_classes,)
        assert min(y) == 0
        assert max(y) == 1

    # check that points were allocated in memory after the first call to __getitem__
    assert dataset.input_data is not None
    assert dataset.input_target is not None
    # check that the correct number of points were allocated in memory
    assert dataset.input_data.size()[0] == num_samples_to_create
    assert dataset.input_target.size()[0] == num_samples_to_create

    # verify that you can getch points outside the num_samples_to_create range
    # (still within the total dataset size range)
    x, y = dataset[num_samples_to_create + 1]
    assert x is not None
    assert y is not None


@pytest.mark.parametrize('label_type', [
    SyntheticDataLabelType.CLASSIFICATION_ONE_HOT,
    SyntheticDataLabelType.CLASSIFICATION_INT,
])
@pytest.mark.parametrize('num_classes', [None, 0])
def test_synthetic_classification_param_validation(label_type: SyntheticDataLabelType, num_classes: Optional[int]):
    with pytest.raises(ValueError):
        SyntheticBatchPairDataset(total_dataset_size=10,
                                  data_shape=(2, 2),
                                  label_type=label_type,
                                  num_classes=num_classes)


@pytest.mark.parametrize('data_type', [
    SyntheticDataType.GAUSSIAN,
    SyntheticDataType.SEPARABLE,
])
@pytest.mark.parametrize('label_type', [
    SyntheticDataLabelType.CLASSIFICATION_ONE_HOT,
    SyntheticDataLabelType.CLASSIFICATION_INT,
])
def test_synthetic_image_data_creation(data_type: SyntheticDataType, label_type: SyntheticDataLabelType):
    if data_type == SyntheticDataType.SEPARABLE:
        if label_type != SyntheticDataLabelType.CLASSIFICATION_INT:
            pytest.skip("Seperable data requires classification int labels")
        num_classes = 2
        label_shape = None
    else:
        num_classes = 10
        label_shape = (1, 10, 12)
        # run run
        return

    dataset_size = 1000
    data_shape = (32, 32)
    num_samples_to_create = 100
    dataset = SyntheticPILDataset(total_dataset_size=dataset_size,
                                  data_shape=data_shape,
                                  num_unique_samples_to_create=num_samples_to_create,
                                  data_type=data_type,
                                  label_type=label_type,
                                  num_classes=num_classes,
                                  label_shape=label_shape)
    assert len(dataset) == dataset_size

    # verify datapoints are correct
    x, y = dataset[0]
    assert x.size == data_shape
    if label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
        assert isinstance(y.item(), int)
    elif label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
        assert y.size() == (num_classes,)
        assert min(y) == 0
        assert max(y) == 1

    # check that points were allocated in memory after the first call to __getitem__
    assert dataset._dataset.input_data is not None
    assert dataset._dataset.input_target is not None
    # check that the correct number of points were allocated in memory
    assert dataset._dataset.input_data.shape[0] == num_samples_to_create
    assert dataset._dataset.input_target.shape[0] == num_samples_to_create

    # verify that you can getch points outside the num_samples_to_create range
    # (still within the total dataset size range)
    x, y = dataset[num_samples_to_create + 1]
    assert x is not None
    assert y is not None
