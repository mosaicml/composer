# Copyright 2021 MosaicML. All Rights Reserved.

"""The `BERT <https://huggingface.co/docs/transformers/master/en/model_doc/bert>`_ model family using `Hugging Face
Transformers <https://huggingface.co/transformers/>`_."""

from composer.models.bert.bert_hparams import BERTForClassificationHparams as BERTForClassificationHparams
from composer.models.bert.bert_hparams import BERTHparams as BERTHparams
from composer.models.bert.model import BERTModel as BERTModel

__all__ = ["BERTModel", "BERTHparams", "BERTForClassificationHparams"]
