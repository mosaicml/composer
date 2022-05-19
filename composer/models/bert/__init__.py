# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The `BERT <https://huggingface.co/docs/transformers/master/en/model_doc/bert>`_ model family using `Hugging Face
Transformers <https://huggingface.co/transformers/>`_."""

from composer.models.bert.bert_hparams import BERTForClassificationHparams as BERTForClassificationHparams
from composer.models.bert.bert_hparams import BERTHparams as BERTHparams
from composer.models.bert.model import BERTModel as BERTModel

__all__ = ["BERTModel", "BERTHparams", "BERTForClassificationHparams"]
