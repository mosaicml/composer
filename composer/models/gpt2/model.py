# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import transformers
from torchmetrics.collections import MetricCollection

from composer.models.nlp_metrics import Perplexity
from composer.models.transformer_shared import MosaicTransformer

if TYPE_CHECKING:
    from composer.core.types import Batch, Metrics, Tensors


class GPT2Model(MosaicTransformer):
    """
    Implements a GPT-2 wrapper around a MosaicTransformer.
    """

    def __init__(self, module: transformers.GPT2Model, config: transformers.GPT2Config, tokenizer_name: str) -> None:
        super().__init__(
            module=module,  #type: ignore (thirdparty)
            config=config,
            tokenizer_name=tokenizer_name)

        # If we ever have algorithms that modify the loss function, then this might be a bit inefficient
        #  because it'll compute the expensive softmax operation twice.
        # Instead, we should consider figuring out how to leverage self.train_loss and return the e^self.train_loss.
        # Of course, this also depends on the implementation details of algorithms.
        self.train_perplexity = Perplexity()
        self.val_perplexity = Perplexity()

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:

        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection([self.train_loss, self.train_perplexity]) if train else MetricCollection(
            [self.val_loss, self.val_perplexity])
