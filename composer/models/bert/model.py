# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

import torch.nn as nn
from torchmetrics import Accuracy, MatthewsCorrcoef, MeanSquaredError, SpearmanCorrcoef
from torchmetrics.collections import MetricCollection

from composer.models.nlp_metrics import BinaryF1Score, CrossEntropyLoss, MaskedAccuracy
from composer.models.transformer_shared import ComposerTransformer

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch, BatchDict, Metrics, Tensors

__all__ = ["BERTModel"]


class BERTModel(ComposerTransformer):
    """Implements a BERT wrapper around a ComposerTransformer.

    See this `paper <https://arxiv.org/abs/1810.04805>`_
    for details on the BERT architecutre.

    Args:
        module (transformers.BertModel): The model to wrap with this module.
        config (transformers.BertConfig): The config for the model.
        tokenizer (transformers.BertTokenizer): The tokenizer used for this model,
            necessary to assert required model inputs.
    """

    def __init__(self, module: transformers.BertModel, config: transformers.BertConfig,
                 tokenizer: transformers.BertTokenizer) -> None:
        super().__init__(
            module=module,  #type: ignore (thirdparty)
            config=config,
            tokenizer=tokenizer)

        # we're going to remove the label from the expected inputs
        # since we will handle metric calculation with TorchMetrics instead of HuggingFace.
        self.model_inputs.remove("labels")

        self.train_metrics = []
        self.val_metrics = []

        # TODO (Moin): make sure this is moved to be dataset-specific
        # if config.num_labels=1, then we are training a regression task, so we should update our loss functions
        if config.num_labels == 1:
            self.train_loss = MeanSquaredError()
            self.val_loss = MeanSquaredError()

            self.train_spearman = SpearmanCorrcoef()
            self.val_spearman = SpearmanCorrcoef()

            self.train_metrics.extend([self.train_loss, self.train_spearman])
            self.val_metrics.extend([self.val_loss, self.val_spearman])

        if config.num_labels == 2:
            self.train_f1 = BinaryF1Score()
            self.val_f1 = BinaryF1Score()

            self.train_metrics.extend([self.train_f1])
            self.val_metrics.extend([self.val_f1])

        if config.num_labels > 1 and config.num_labels != len(self.tokenizer):
            self.train_acc = Accuracy()
            self.val_acc = Accuracy()

            self.train_matthews = MatthewsCorrcoef(num_classes=config.num_labels)
            self.val_matthews = MatthewsCorrcoef(num_classes=config.num_labels)

            self.train_metrics.extend([self.train_acc, self.train_matthews])
            self.val_metrics.extend([self.val_acc, self.val_matthews])

        if config.num_labels == len(self.tokenizer):  # tests for MLM pre-training
            ignore_index = -100
            self.train_loss = CrossEntropyLoss(ignore_index=ignore_index, vocab_size=config.num_labels)
            self.val_loss = CrossEntropyLoss(ignore_index=ignore_index, vocab_size=config.num_labels)

            self.train_acc = MaskedAccuracy(ignore_index=ignore_index)
            self.val_acc = MaskedAccuracy(ignore_index=ignore_index)

            self.train_metrics.extend([self.train_loss, self.train_acc])
            self.val_metrics.extend([self.val_loss, self.val_acc])

        if config.num_labels == 1:
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch: BatchDict):
        if not isinstance(batch, dict):
            raise ValueError(f'Model expects batch to be a dict, got {type(batch)}')

        for key in self.model_inputs:
            if key not in batch.keys():
                raise ValueError(f'Batch missing key: {key}')

        labels = batch.pop('labels')
        output = self.module(**batch)  # type: ignore (thirdparty)
        batch['labels'] = labels
        return output

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        logits = outputs['logits']
        labels = batch['labels']
        loss = self.loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def validate(self, batch: BatchDict) -> Tuple[Tensors, Tensors]:
        """Runs the validation step.

        Args:
            batch (BatchDict): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in ComposerTransformer.get_model_inputs().

        Returns:
            A tuple of (Tensor, Tensor) with the output from the forward pass and the correct labels.
            This is fed into directly into the output of :meth:`metrics`.
        """

        assert self.training is False, "For validation, model must be in eval mode"

        output = self.forward(batch)
        output = output['logits']
        labels = batch['labels']

        # if we are in the single class case, then remove the classes dimension
        if output.shape[1] == 1:
            output = output.squeeze(dim=1)

        return (output, labels)

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection(self.train_metrics) if train else MetricCollection(self.val_metrics)
