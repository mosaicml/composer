# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest

from composer.core.state import State
from composer.datasets.dataset_hparams import DataLoaderHparams
from composer.datasets.lm_dataset_hparams import LMDatasetHparams
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder
from composer.models import BERTHparams, GPT2Hparams, create_bert_mlm, create_gpt2
from tests.common.models import generate_dummy_model_config
from tests.datasets import test_synthetic_lm_data


def make_dataset_configs(model_family=('bert', 'gpt2')) -> list:
    model_family = list(model_family)
    lm_dataset_configs = [
        config[0] for config in test_synthetic_lm_data.generate_parameter_configs(
            ['num_samples', 'chars_per_sample', 'column_names', 'tokenizer_family'], model_family=model_family)
    ]
    return lm_dataset_configs


def make_lm_tokenizer(config: dict):
    pytest.importorskip('transformers')
    dataset = synthetic_hf_dataset_builder(num_samples=config['num_samples'],
                                           chars_per_sample=config['chars_per_sample'],
                                           column_names=config['column_names'])
    tokenizer = generate_synthetic_tokenizer(config['tokenizer_family'], dataset)
    return tokenizer


def make_dummy_lm(model_name: str, max_position_embeddings: int, tokenizer):
    if model_name == 'gpt2':
        class_name = GPT2Hparams
        model_config = generate_dummy_model_config(class_name, tokenizer)
        model_config['max_position_embeddings'] = max_position_embeddings
        model = create_gpt2(model_config=model_config)
    elif model_name == 'bert':
        class_name = BERTHparams
        model_config = generate_dummy_model_config(class_name, tokenizer)
        model_config['max_position_embeddings'] = max_position_embeddings
        model = create_bert_mlm(model_config=model_config)
    else:
        raise ValueError("Model name must be one of 'gpt2' or 'bert'")
    return model


def make_synthetic_dataloader(dataset_config: dict):
    """creates a dataloader for synthetic sequence data."""
    pytest.importorskip('transformers')
    dataloader = LMDatasetHparams(use_synthetic=True,
                                  tokenizer_name=dataset_config['tokenizer_family'],
                                  use_masked_lm=dataset_config['use_masked_lm'],
                                  max_seq_length=dataset_config['chars_per_sample'],
                                  split='train')
    dataloader = dataloader.initialize_object(batch_size=dataset_config['num_samples'],
                                              dataloader_hparams=DataLoaderHparams(num_workers=0,
                                                                                   persistent_workers=False))
    return dataloader


def make_synthetic_model(config):
    tokenizer = make_lm_tokenizer(config)
    model = make_dummy_lm(config['tokenizer_family'], config['chars_per_sample'], tokenizer)
    return model


def make_synthetic_bert_model():
    config = make_dataset_configs(model_family=['bert'])[0]
    return make_synthetic_model(config)


def make_synthetic_bert_dataloader():
    config = make_dataset_configs(model_family=['bert'])[0]
    return make_synthetic_dataloader(config)


def make_synthetic_gpt2_model():
    config = make_dataset_configs(model_family=['gpt2'])[0]
    return make_synthetic_model(config)


def make_synthetic_gpt2_dataloader():
    config = make_dataset_configs(model_family=['gpt2'])[0]
    return make_synthetic_dataloader(config)


def synthetic_hf_state_maker(config) -> Tuple:
    """An example state using synthetic HF transformer function which could used for testing purposes."""
    model = make_synthetic_model(config)
    dataloader = make_synthetic_dataloader(config)
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
    )

    return state, model, dataloader


@pytest.fixture(params=make_dataset_configs())
def synthetic_hf_state(request):
    pytest.importorskip('transformers')
    config = request.param
    return synthetic_hf_state_maker(config)
