# Copyright 2021 MosaicML. All Rights Reserved.

from itertools import product
from typing import Optional

import pytest

from composer.datasets.synthetic import (SyntheticBatchPairDataset, SyntheticDataLabelType, SyntheticDataType,
                                         SyntheticPILDataset)
from composer.datasets.synthetic_lm import SyntheticHFDataset, generate_synthetic_tokenizer


def generate_parameter_configs(keys, num_replicas=1):
    config_options = {
        "tokenizer_family": ['bert', 'gpt2'],
        "chars_per_sample": [128],
        "column_names": [['sentence'], ['sentence1', 'sentence2']],
        "num_samples": [50]
    }

    config_combinations = []
    for combo in product(*[config_options[i] for i in keys]):
        config_combinations.append([dict(zip(keys, combo)) for _ in range(num_replicas)])
    return config_combinations


@pytest.fixture
def config(request):
    return request.param


@pytest.fixture
def dataset_generator(request):
    print(request.param)
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    pytest.importorskip("tokenizers")

    dataset_generator = SyntheticHFDataset(num_samples=request.param['num_samples'],
                                           chars_per_sample=request.param['chars_per_sample'],
                                           column_names=request.param['column_names'])
    return dataset_generator


@pytest.fixture
def dataset(dataset_generator):
    return dataset_generator.generate_dataset()


@pytest.mark.parametrize("dataset_generator, config",
                         generate_parameter_configs(['num_samples', 'chars_per_sample', 'column_names'],
                                                    num_replicas=2),
                         indirect=True)
def test_generator_sample(dataset_generator, config):
    sample = dataset_generator.generate_sample()
    assert len(sample) == config['chars_per_sample']


@pytest.mark.parametrize("dataset_generator, config",
                         generate_parameter_configs(['num_samples', 'chars_per_sample', 'column_names'],
                                                    num_replicas=2),
                         indirect=True)
def test_dataset_properties(dataset, config):
    assert len(dataset) == config['num_samples']
    assert len(dataset[config['column_names'][0]][0]) == config['chars_per_sample']
    assert dataset.column_names == (config['column_names'] + ['idx'])


@pytest.fixture
def tokenizer(dataset, config):
    # build the tokenizer
    tokenizer = generate_synthetic_tokenizer(config['tokenizer_family'], dataset=dataset)
    # verifying the input ids are a part of the tokenizer
    assert 'input_ids' in tokenizer.model_input_names
    return tokenizer


@pytest.fixture
def tokenized_dataset(tokenizer, dataset, config):
    # test tokenizing the dataset
    max_length = config['chars_per_sample'] * 2
    dataset = dataset.map(lambda inp: tokenizer(
        text=inp[config['column_names'][0]], padding="max_length", max_length=max_length, truncation=True),
                          batched=True,
                          num_proc=1,
                          keep_in_memory=True)
    return dataset


@pytest.mark.parametrize("dataset_generator, config",
                         generate_parameter_configs(
                             ['num_samples', 'chars_per_sample', 'column_names', 'tokenizer_family'], num_replicas=2),
                         indirect=True)
def test_tokenizer_specific_properties(tokenizer, tokenized_dataset, config):
    pytest.importorskip("transformers")
    from transformers import BertTokenizer, GPT2Tokenizer

    # verify datapoints are correct
    assert 'input_ids' in tokenized_dataset.column_names
    x = tokenized_dataset['input_ids'][0]
    max_length = config['chars_per_sample'] * 2
    assert len(x) == max_length

    # add some tokenizer-specific tests
    if config['tokenizer_family'] == "bert":
        assert x[0] == tokenizer.cls_token_id
        assert tokenizer.sep_token_id in x

    # since our tokenization max_length==chars_per_sample, we should always have padding tokens due to extra space
    assert x[-1] == tokenizer.pad_token_id

    if config['tokenizer_family'] == "bert":
        assert isinstance(tokenizer, BertTokenizer)
    elif config['tokenizer_family'] == "gpt2":
        assert isinstance(tokenizer, GPT2Tokenizer)

    assert tokenizer.pad_token_id == 0
    if config['tokenizer_family'] == "bert":
        assert tokenizer.cls_token is not None
        assert tokenizer.sep_token is not None
    elif config['tokenizer_family'] == "gpt2":
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
