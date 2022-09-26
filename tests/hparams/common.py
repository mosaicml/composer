# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import pytest
import yahp as hp
import yaml
from torch.utils.data import RandomSampler, SequentialSampler

from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.glue_hparams import GLUEHparams
from composer.datasets.lm_dataset_hparams import LMDatasetHparams
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer
from composer.models import ModelHparams
from composer.models.bert.bert_hparams import BERTForClassificationHparams, BERTHparams
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabV3Hparams
from composer.models.gpt2.gpt2_hparams import GPT2Hparams
from composer.models.model_hparams import ModelHparams
from tests.common import RandomClassificationDataset, SimpleConvModel, SimpleModel

T = TypeVar('T')


def assert_in_registry(constructor: Callable, registry: Dict[str, Callable]):
    """Assert that the ``registry`` contains ``constructor``."""
    registry_entries = set(registry.values())
    assert constructor in registry_entries, f'Constructor {constructor.__name__} is missing from the registry.'


def construct_from_yaml(
    constructor: Callable[..., T],
    yaml_dict: Optional[Dict[str, Any]] = None,
) -> T:
    """Build ``constructor`` from ``yaml_dict``

    Args:
        constructor (Callable): The constructor to test (such as an Hparams class)
        yaml_dict (Dict[str, Any], optional): The YAML. Defaults to ``None``, which is equivalent
            to an empty dictionary.
    """
    yaml_dict = {} if yaml_dict is None else yaml_dict
    # ensure that yaml_dict is actually a dictionary of only json-serializable objects
    yaml_dict = yaml.safe_load(yaml.safe_dump(yaml_dict))
    instance = hp.create(constructor, yaml_dict, cli_args=False)
    return instance


model_hparams_to_tokenizer_family: Dict[Type[ModelHparams], str] = {
    GPT2Hparams: 'gpt2',
    BERTForClassificationHparams: 'bert',
    BERTHparams: 'bert'
}


@dataclasses.dataclass
class RandomClassificationDatasetHparams(DatasetHparams, SyntheticHparamsMixin):

    data_shape: List[int] = hp.optional('data shape', default_factory=lambda: [1, 1, 1])
    num_classes: int = hp.optional('num_classes', default=2)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):
        assert self.data_shape is not None
        assert self.num_classes is not None
        dataset = RandomClassificationDataset(
            size=self.synthetic_num_unique_samples,
            shape=self.data_shape,
            num_classes=self.num_classes,
        )
        if self.shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
        )


def configure_dataset_hparams_for_synthetic(
    dataset_hparams: DatasetHparams,
    model_hparams: Optional[ModelHparams] = None,
) -> None:
    if not isinstance(dataset_hparams, SyntheticHparamsMixin):
        pytest.xfail(f'{dataset_hparams.__class__.__name__} does not support synthetic data or num_total_batches')

    assert isinstance(dataset_hparams, SyntheticHparamsMixin)

    dataset_hparams.use_synthetic = True

    if model_hparams and type(model_hparams) in model_hparams_to_tokenizer_family:
        tokenizer_family = model_hparams_to_tokenizer_family[type(model_hparams)]
        assert isinstance(dataset_hparams, (GLUEHparams, LMDatasetHparams))
        dataset_hparams.tokenizer_name = tokenizer_family
        dataset_hparams.max_seq_length = 128


@dataclasses.dataclass
class SimpleModelHparams(ModelHparams):
    num_features: int = hp.optional('number of features', default=1)
    num_classes: int = hp.optional('number of output classes', default=2)

    def initialize_object(self) -> SimpleModel:
        return SimpleModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
        )


@dataclasses.dataclass
class SimpleConvModelHparams(ModelHparams):
    num_channels: int = hp.optional('number of channels', default=3)
    num_classes: int = hp.optional('number of output classes', default=2)

    def initialize_object(self) -> SimpleConvModel:
        return SimpleConvModel(
            num_channels=self.num_channels,
            num_classes=self.num_classes,
        )


def configure_model_hparams_for_synthetic(model_hparams: ModelHparams) -> None:
    # configure Transformer-based models for synthetic testing
    if type(model_hparams) in model_hparams_to_tokenizer_family.keys():
        assert isinstance(model_hparams, (BERTHparams, GPT2Hparams, BERTForClassificationHparams))
        tokenizer_family = model_hparams_to_tokenizer_family[type(model_hparams)]

        # force a non-pretrained model
        model_hparams.use_pretrained = False
        model_hparams.pretrained_model_name = None

        # generate tokenizers and synthetic models
        tokenizer = generate_synthetic_tokenizer(tokenizer_family=tokenizer_family)
        model_hparams.model_config = generate_dummy_model_config(type(model_hparams), tokenizer)

    # configure DeepLabV3 models for synthetic testing
    if isinstance(model_hparams, DeepLabV3Hparams):
        model_hparams.backbone_weights = None  # prevent downloading pretrained weights during test
        model_hparams.sync_bn = False  # sync_bn throws an error when run on CPU


def generate_dummy_model_config(cls: Type[hp.Hparams], tokenizer) -> Dict[str, Any]:
    model_to_dummy_mapping: Dict[Type[hp.Hparams], Dict[str, Any]] = {
        BERTHparams: {
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
        GPT2Hparams: {
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
        BERTForClassificationHparams: {
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
    return model_to_dummy_mapping[cls]
