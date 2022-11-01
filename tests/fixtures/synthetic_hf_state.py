# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Tuple

import pytest

from composer.core.state import State
from composer.datasets.lm_dataset import build_synthetic_lm_dataloader
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder
from composer.models import create_bert_mlm, create_gpt2
from tests.datasets import test_synthetic_lm_data


def generate_dummy_model_config(model: str, tokenizer) -> Dict[str, Any]:
    model_to_dummy_mapping: Dict[str, Dict[str, Any]] = {
        'bert': {
            'architectures': ['BertForMaskedLM'],
            'attention_probs_dropout_prob': 0.1,
            'gradient_checkpointing': False,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'hidden_size': 64,
            'initializer_range': 0.02,
            'intermediate_size': 256,
            'layer_norm_eps': 1e-12,
            'max_position_embeddings': 512,
            'model_type': 'bert',
            'num_attention_heads': 1,
            'num_hidden_layers': 1,
            'pad_token_id': tokenizer.pad_token_id,
            'position_embedding_type': 'absolute',
            'transformers_version': '4.6.0.dev0',
            'type_vocab_size': 2,
            'use_cache': True,
            'vocab_size': tokenizer.vocab_size,
        },
        'gpt2': {
            'activation_function': 'gelu_new',
            'architectures': ['GPT2LMHeadModel'],
            'attn_pdrop': 0.1,
            'bos_token_id': tokenizer.cls_token_id,
            'embd_pdrop': 0.1,
            'eos_token_id': tokenizer.cls_token_id,
            'initializer_range': 0.02,
            'layer_norm_epsilon': 0.00001,
            'model_type': 'gpt2',
            'n_ctx': 128,
            'n_embd': 64,
            'n_head': 1,
            'n_layer': 1,
            'n_positions': 128,
            'resid_pdrop': 0.1,
            'summary_activation': None,
            'summary_first_dropout': 0.1,
            'summary_proj_to_labels': True,
            'summary_type': 'cls_index',
            'summary_use_proj': True,
            'task_specific_params': {
                'text-generation': {
                    'do_sample': True,
                    'max_length': 50
                }
            },
            'vocab_size': tokenizer.vocab_size
        },
        'bert_classification': {
            'architectures': ['BertForSequenceClassification'],
            'attention_probs_dropout_prob': 0.1,
            'classifier_dropout': None,
            'gradient_checkpointing': False,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'hidden_size': 64,
            'initializer_range': 0.02,
            'intermediate_size': 256,
            'layer_norm_eps': 1e-12,
            'max_position_embeddings': 512,
            'model_type': 'bert',
            'num_attention_heads': 1,
            'num_hidden_layers': 1,
            'pad_token_id': tokenizer.pad_token_id,
            'position_embedding_type': 'absolute',
            'transformers_version': '4.16.2',
            'type_vocab_size': 2,
            'use_cache': True,
            'vocab_size': tokenizer.vocab_size
        }
    }
    return model_to_dummy_mapping[model]


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
        model_config = generate_dummy_model_config(model_name, tokenizer)
        model_config['max_position_embeddings'] = max_position_embeddings
        model = create_gpt2(model_config=model_config)
    elif model_name == 'bert':
        model_config = generate_dummy_model_config(model_name, tokenizer)
        model_config['max_position_embeddings'] = max_position_embeddings
        model = create_bert_mlm(model_config=model_config)
    else:
        raise ValueError("Model name must be one of 'gpt2' or 'bert'")
    return model


def make_synthetic_dataloader(dataset_config: dict):
    """creates a dataloader for synthetic sequence data."""
    pytest.importorskip('transformers')
    return build_synthetic_lm_dataloader(
        synthetic_num_unique_samples=100,
        batch_size=dataset_config['num_samples'],
        tokenizer_name=dataset_config['tokenizer_family'],
        use_masked_lm=dataset_config['use_masked_lm'],
        max_seq_length=dataset_config['chars_per_sample'],
        split='train',
    )


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
