# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 model based on `Hugging Face GPT-2 <https://huggingface.co/docs/transformers/master/en/model_doc/gpt2>`_.

Implemented as a wrapper using :class:`.ComposerTrainer`.
"""

from __future__ import annotations

from typing import Optional

from composer.metrics.nlp import HFCrossEntropy, Perplexity
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['create_gpt2']


def create_gpt2(use_pretrained: Optional[bool] = False,
                pretrained_model_name: Optional[str] = None,
                model_config: Optional[dict] = None,
                tokenizer_name: Optional[str] = None,
                gradient_checkpointing: Optional[bool] = False):
    """Implements :class:`~composer.models.huggingface.HuggingFaceModel` to wrap `Hugging Face GPT-2 \
    transformers <https://huggingface.co/docs/transformers/master/en/model_doc/gpt2#overview>`_. Logs training and
    validation perplexity.

    From `Language Models are Unsupervised Multitask Learners <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_ (Radford et al, 2018).

    Args:

        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): A dictionary providing a HuggingFace model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset
        and validate the models inputs.

        .. code-block::

            {
              "_name_or_path": "gpt2",
              "activation_function": "gelu_new",
              "architectures": ["GPT2LMHeadModel"],
              "attn_pdrop": 0.1,
              "bos_token_id": 50256,
              "embd_pdrop": 0.1,
              "eos_token_id": 50256,
              "initializer_range": 0.02,
              "layer_norm_epsilon": 1e-05,
              "model_type": "gpt2",
              "n_ctx": 1024,
              "n_embd": 768,
              "n_head": 12,
              "n_inner": null,
              "n_layer": 12,
              "n_positions": 1024,
              "reorder_and_upcast_attn": false,
              "resid_pdrop": 0.1,
              "scale_attn_by_inverse_layer_idx": false,
              "scale_attn_weights": true,
              "summary_activation": null,
              "summary_first_dropout": 0.1,
              "summary_proj_to_labels": true,
              "summary_type": "cls_index",
              "summary_use_proj": true,
              "task_specific_params": {
              "text-generation": {
              "do_sample": true,
              "max_length": 50 }
              },
              "transformers_version": "4.16.0",
              "use_cache": true,
              "vocab_size": 50257
            }

   To create a GPT-2 model for language modeling pretraining:

    .. testcode::

        from composer.models import create_gpt2

        composer_model = create_gpt2()

    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'gpt2'

    if use_pretrained:
        assert transformers.AutoModelForCausalLM.from_pretrained is not None, 'AutoModelForCausalLM has from_pretrained method'
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,
                                                                  **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name, **model_config)
        assert transformers.AutoModelForCausalLM.from_config is not None, 'AutoModelForCausalLM has from_config method'
        model = transformers.AutoModelForCausalLM.from_config(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    return HuggingFaceModel(model=model, tokenizer=tokenizer, metrics=[HFCrossEntropy(), Perplexity()])
