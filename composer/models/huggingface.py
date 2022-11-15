# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts 🤗 Transformers models to composer models"""

from __future__ import annotations

import logging
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from torchmetrics import Metric

from composer.models.base import ComposerModel

if TYPE_CHECKING:
    import transformers

log = logging.getLogger(__name__)

__all__ = ['HuggingFaceModel']


class HuggingFaceModel(ComposerModel):
    """
    A wrapper class that converts 🤗 Transformers models to composer models.

    Args:
        model (transformers.PreTrainedModel): A 🤗 Transformers model.
        tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer used to prepare the dataset. Default ``None``.
        use_logits (bool, optional): If True, the model's output logits will be used to calculate validation metrics. Else, metrics will be inferred from the HuggingFaceModel directly. Default: ``False``
        metrics (list[Metric], optional): list of torchmetrics to apply to the output of `validate`. Default: ``None``.
    .. warning:: This wrapper is designed to work with 🤗 datasets that define a `labels` column.

    Example:

    .. testcode::

        import transformers
        from composer.models import HuggingFaceModel

        hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model = HuggingFaceModel(hf_model)
    """

    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: Optional[Union[transformers.PreTrainedTokenizer,
                                           transformers.PreTrainedTokenizerFast]] = None,
                 use_logits: Optional[bool] = False,
                 metrics: Optional[List[Metric]] = None) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.tokenizer = tokenizer

        if tokenizer is not None and self.config.vocab_size != len(tokenizer):
            # set model's word embedding matrix and final lm_head to vocab size according to tokenizer
            log.warning(f'The number of tokens in the tokenizer and the number of tokens in the model are different.'
                        f' Resizing the model tokenizer to {len(tokenizer)} from {self.config.vocab_size}.')
            self.model.resize_token_embeddings(len(tokenizer))

        self.use_logits = use_logits

        self.train_metrics = None
        self.val_metrics = None

        if metrics:
            self.train_metrics = {metric.__class__.__name__: metric for metric in metrics}
            self.val_metrics = {metric.__class__.__name__: metric for metric in metrics}

        self.labels = None  # set in eval_forward() if exists

    def forward(self, batch):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
            )
        return output

    def loss(self, outputs, batch):
        if self.config.use_return_dict:
            return outputs['loss']
        else:
            # loss is at index 0 in the output tuple
            return outputs[0]

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        output = outputs if outputs else self.forward(batch)
        if self.use_logits:
            self.labels = batch.pop('labels')
            if self.config.use_return_dict:
                output = output['logits']
            else:
                # logits are at index 1 in the output tuple
                output = output[1]

            # if we are in the single class case, then remove the classes dimension
            if output.shape[1] == 1:
                output = output.squeeze(dim=1)

        return output

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        return metrics if metrics else {}

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        metric.update(outputs, self.labels)
