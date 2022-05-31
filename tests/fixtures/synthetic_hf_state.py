# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

#from collections.abc import Iterable

import pytest
from transformers import PreTrainedTokenizer

from composer.core.state import State
from composer.datasets import DataLoaderHparams, LMDatasetHparams
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder
from composer.models import BERTHparams, GPT2Hparams, TransformerHparams
from tests.common.models import generate_dummy_model_config
from tests.datasets import test_synthetic_lm_data


def make_dataset_configs() -> list:
    lm_dataset_configs = [
        config[0] for config in test_synthetic_lm_data.generate_parameter_configs(
            ['num_samples', 'chars_per_sample', 'column_names', 'tokenizer_family'])
    ]
    return lm_dataset_configs


def make_lm_tokenizer(config: dict):
    dataset = synthetic_hf_dataset_builder(num_samples=config['num_samples'],
                                           chars_per_sample=config['chars_per_sample'],
                                           column_names=config['column_names'])
    tokenizer = generate_synthetic_tokenizer(config['tokenizer_family'], dataset)
    return tokenizer


def make_dummy_lm(model_name: str, max_position_embeddings: int, tokenizer: PreTrainedTokenizer):
    class_name = TransformerHparams
    if model_name == 'gpt2':
        class_name = GPT2Hparams
    elif model_name == 'bert':
        class_name = BERTHparams
    else:
        raise ValueError("Model name must be one of 'gpt2' or 'bert'")
    model_config = generate_dummy_model_config(class_name, tokenizer)
    model_config['max_position_embeddings'] = max_position_embeddings
    model = class_name(model_config=model_config).initialize_object()
    return model


def make_synthetic_dataloader(dataset_config: dict):
    """
    """
    dataloader = LMDatasetHparams(use_synthetic=True,
                                  tokenizer_name=dataset_config['tokenizer_family'],
                                  use_masked_lm=dataset_config['use_masked_lm'],
                                  max_seq_length=dataset_config["chars_per_sample"],
                                  split='train')
    dataloader = dataloader.initialize_object(batch_size=dataset_config['num_samples'],
                                              dataloader_hparams=DataLoaderHparams(num_workers=0,
                                                                                   persistent_workers=False))
    return dataloader


def model_components(config):
    tokenizer = make_lm_tokenizer(config)
    model = make_dummy_lm(config['tokenizer_family'], config['chars_per_sample'], tokenizer)
    dataloader = make_synthetic_dataloader(config)
    return model, dataloader


def synthetic_hf_state_maker(config) -> State:
    """
    An example state using synthetic HF transformer function which could used for testing purposes
    """
    #config = request.params
    model, dataloader = model_components(config)
    state = State(
        model=model,
        rank_zero_seed=0,
        dataloader=dataloader,
        dataloader_label="train",
        max_duration='1ep',
    )

    return state


@pytest.fixture(params=make_dataset_configs())
def synthetic_hf_state_fixture(request):
    config = request.param
    return synthetic_hf_state_maker(config)
