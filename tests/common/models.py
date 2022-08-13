# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Contains commonly used models that are shared across the test suite."""

import dataclasses
from typing import Any, Dict, Type

import torch
import yahp as hp

from composer.datasets.synthetic_lm import generate_synthetic_tokenizer
from composer.models import ComposerClassifier
from composer.models.bert.bert_hparams import BERTForClassificationHparams, BERTHparams
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabV3Hparams
from composer.models.gpt2.gpt2_hparams import GPT2Hparams
from composer.models.model_hparams import ModelHparams

model_hparams_to_tokenizer_family: Dict[Type[ModelHparams], str] = {
    GPT2Hparams: 'gpt2',
    BERTForClassificationHparams: 'bert',
    BERTHparams: 'bert'
}


class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 1, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, 5)
        fc2 = torch.nn.Linear(5, num_classes)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = fc1
        self.fc2 = fc2


@dataclasses.dataclass
class SimpleModelHparams(ModelHparams):
    num_features: int = hp.optional('number of features', default=1)
    num_classes: int = hp.optional('number of output classes', default=2)

    def initialize_object(self) -> SimpleModel:
        return SimpleModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
        )


class SimpleConvModel(ComposerClassifier):
    """Small convolutional classifer.

    Args:
        num_channels (int): number of input channels (default: 3)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 2) -> None:

        self.num_classes = num_classes
        self.num_channels = num_channels

        conv_args = {'kernel_size': (3, 3), 'padding': 1, 'stride': 2}
        conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=8, **conv_args)
        conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, **conv_args)
        pool = torch.nn.AdaptiveAvgPool2d(1)
        flatten = torch.nn.Flatten()
        fc1 = torch.nn.Linear(4, 16)
        fc2 = torch.nn.Linear(16, num_classes)

        net = torch.nn.Sequential(
            conv1,
            conv2,
            pool,
            flatten,
            fc1,
            fc2,
        )
        super().__init__(module=net)

        # bind these to class for access during
        # surgery tests
        self.conv1 = conv1
        self.conv2 = conv2


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
