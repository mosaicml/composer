# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a BERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

from typing import Optional

from torchmetrics import MeanSquaredError
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

from composer.metrics.nlp import BinaryF1Score, LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['create_bert_mlm', 'create_bert_classification']


def create_bert_mlm(use_pretrained: Optional[bool] = False,
                    pretrained_model_name: Optional[str] = None,
                    model_config: Optional[dict] = None,
                    tokenizer_name: Optional[str] = None,
                    gradient_checkpointing: Optional[bool] = False):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:

        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
        architecture of a Hugging Face model.
        tokenizer_name (transformers.BertTokenizer, optional): Tokenizer name used to preprocess the dataset
        and validate the models inputs.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": ["BertForMaskedLM"],
              "attention_probs_dropout_prob": 0.1,
              "classifier_dropout": null,
              "gradient_checkpointing": false,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "layer_norm_eps": 1e-12,
              "max_position_embeddings": 512,
              "model_type": "bert",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 0,
              "position_embedding_type": "absolute",
              "transformers_version": "4.16.0",
              "type_vocab_size": 2,
              "use_cache": true,
              "vocab_size": 30522
            }

   To create a BERT model for Masked Language Model pretraining:

    .. testcode::

        from composer.models import create_bert_mlm
        model = create_bert_mlm()

    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    if use_pretrained:
        assert transformers.AutoModelForMaskedLM.from_pretrained is not None, 'AutoModelForMaskedLM has from_pretrained method'
        model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,
                                                                  **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name, **model_config)
        assert transformers.AutoModelForMaskedLM.from_config is not None, 'AutoModelForMaskedLM has from_config method'
        model = transformers.AutoModelForMaskedLM.from_config(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=model.config.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    return HuggingFaceModel(model=model, tokenizer=tokenizer, use_logits=True, metrics=metrics)


def create_bert_classification(num_labels: Optional[int] = 2,
                               use_pretrained: Optional[bool] = False,
                               pretrained_model_name: Optional[str] = None,
                               model_config: Optional[dict] = None,
                               tokenizer_name: Optional[str] = None,
                               gradient_checkpointing: Optional[bool] = False):
    """BERT classification model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        num_labels (int, optional): The number of classes in the classification task. Default: ``2``.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
        architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset
        and validate the models inputs.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": [
                "BertForSequenceClassification
              ],
              "attention_probs_dropout_prob": 0.1,
              "classifier_dropout": null,
              "gradient_checkpointing": false,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1",
                "2": "LABEL_2"
              },
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2
              },
              "layer_norm_eps": 1e-12,
              "max_position_embeddings": 512,
              "model_type": "bert",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 0,
              "position_embedding_type": "absolute",
              "transformers_version": "4.16.0",
              "type_vocab_size": 2,
              "use_cache": true,
              "vocab_size": 30522
            }

   To create a BERT model for classification:

    .. testcode::

        from composer.models import create_bert_classification
        model = create_bert_classification(num_labels=3) # if the task has three classes.

    Note:
        This function can be used to construct a BERT model for regression by setting ``num_labels == 1``.
        This will have two noteworthy effects. First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics will be :class:`~torchmetrics.MeanSquaredError` and :class:`~torchmetrics.SpearmanCorrCoef`.

        For the classification case (when ``num_labels > 1``), the training loss is :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.Accuracy` and :class:`~torchmetrics.MatthewsCorrCoef`, as well as :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    if not model_config:
        model_config = {}

    model_config['num_labels'] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    if use_pretrained:
        assert transformers.AutoModelForSequenceClassification.from_pretrained is not None, 'AutoModelForSequenceClassification has from_pretrained method'
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name, **model_config)
        assert transformers.AutoModelForSequenceClassification.from_config is not None, 'AutoModelForSequenceClassification has from_config method'
        model = transformers.AutoModelForSequenceClassification.from_config(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [Accuracy(), MatthewsCorrCoef(num_classes=model.config.num_labels)]
        if num_labels == 2:
            metrics.append(BinaryF1Score())

    return HuggingFaceModel(model=model, tokenizer=tokenizer, use_logits=True, metrics=metrics)
