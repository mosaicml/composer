# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The `BERT <https://huggingface.co/docs/transformers/master/en/model_doc/bert>`_ model family using `Hugging Face
Transformers <https://huggingface.co/transformers/>`_."""

from composer.models.bert.model import create_bert_classification as create_bert_classification
from composer.models.bert.model import create_bert_mlm as create_bert_mlm

__all__ = ['create_bert_classification', 'create_bert_mlm']
