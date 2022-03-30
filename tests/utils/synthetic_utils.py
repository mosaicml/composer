from typing import Any, Dict, Optional, Type

import pytest

from composer.datasets import GLUEHparams, LMDatasetHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer
from composer.models import (BERTForClassificationHparams, BERTHparams, DeepLabV3Hparams, GPT2Hparams, ModelHparams,
                             TransformerHparams)


def configure_dataset_for_synthetic(dataset_hparams: DatasetHparams,
                                    model_hparams: Optional[ModelHparams] = None) -> None:
    if not isinstance(dataset_hparams, SyntheticHparamsMixin):
        pytest.xfail(f"{dataset_hparams.__class__.__name__} does not support synthetic data or num_total_batches")

    assert isinstance(dataset_hparams, SyntheticHparamsMixin)

    dataset_hparams.use_synthetic = True

    if isinstance(model_hparams, TransformerHparams):
        if type(model_hparams) not in _model_hparams_to_tokenizer_family:
            raise ValueError(f"Model {type(model_hparams)} is currently not supported for synthetic testing!")

        tokenizer_family = _model_hparams_to_tokenizer_family[type(model_hparams)]
        assert isinstance(dataset_hparams, (GLUEHparams, LMDatasetHparams))
        dataset_hparams.tokenizer_name = tokenizer_family
        dataset_hparams.max_seq_length = 128


_model_hparams_to_tokenizer_family: Dict[Type[TransformerHparams], str] = {
    GPT2Hparams: "gpt2",
    BERTForClassificationHparams: "bert",
    BERTHparams: "bert"
}


def configure_model_for_synthetic(model_hparams: ModelHparams) -> None:
    # configure Transformer-based models for synthetic testing
    if isinstance(model_hparams, TransformerHparams):
        if type(model_hparams) not in _model_hparams_to_tokenizer_family:
            raise ValueError(f"Model {type(model_hparams)} is currently not supported for synthetic testing!")

        tokenizer_family = _model_hparams_to_tokenizer_family[type(model_hparams)]

        # force a non-pretrained model
        model_hparams.use_pretrained = False
        model_hparams.pretrained_model_name = None

        # generate tokenizers and synthetic models
        tokenizer = generate_synthetic_tokenizer(tokenizer_family=tokenizer_family)
        model_hparams.tokenizer_name = None
        model_hparams.model_config = generate_dummy_model_config(type(model_hparams), tokenizer)

    # configure DeepLabV3 models for synthetic testing
    if isinstance(model_hparams, DeepLabV3Hparams):
        model_hparams.is_backbone_pretrained = False  # prevent downloading pretrained weights during test
        model_hparams.sync_bn = False  # sync_bn throws an error when run on CPU


def generate_dummy_model_config(class_name, tokenizer) -> Dict[str, Any]:
    model_to_dummy_mapping = {
        BERTHparams: {
            "architectures": ["BertForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "pad_token_id": tokenizer.pad_token_id,
            "position_embedding_type": "absolute",
            "transformers_version": "4.6.0.dev0",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": tokenizer.vocab_size,
        },
        GPT2Hparams: {
            "activation_function": "gelu_new",
            "architectures": ["GPT2LMHeadModel"],
            "attn_pdrop": 0.1,
            "bos_token_id": tokenizer.cls_token_id,
            "embd_pdrop": 0.1,
            "eos_token_id": tokenizer.cls_token_id,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 0.00001,
            "model_type": "gpt2",
            "n_ctx": 128,
            "n_embd": 64,
            "n_head": 1,
            "n_layer": 1,
            "n_positions": 128,
            "resid_pdrop": 0.1,
            "summary_activation": None,
            "summary_first_dropout": 0.1,
            "summary_proj_to_labels": True,
            "summary_type": "cls_index",
            "summary_use_proj": True,
            "task_specific_params": {
                "text-generation": {
                    "do_sample": True,
                    "max_length": 50
                }
            },
            "vocab_size": tokenizer.vocab_size
        },
        BERTForClassificationHparams: {
            "architectures": ["BertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.1,
            "classifier_dropout": None,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "pad_token_id": tokenizer.pad_token_id,
            "position_embedding_type": "absolute",
            "transformers_version": "4.16.2",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": tokenizer.vocab_size
        }
    }
    return model_to_dummy_mapping[class_name]
