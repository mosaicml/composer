# Copyright 2021 MosaicML. All Rights Reserved.

"""Implements a BERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Union

import torch
from torchmetrics import MeanSquaredError, Metric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

from composer.metrics.nlp import BinaryF1Score, LanguageCrossEntropy, MaskedAccuracy
from composer.models.transformer_shared import ComposerTransformer

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch, BatchDict, BatchPair

__all__ = ["BERTModel"]


class BERTModel(ComposerTransformer):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        module (transformers.BertModel): An instance of BertModel that
            contains the forward pass function.
        config (transformers.BertConfig): The BertConfig object that
            stores information about the model hyperparameters.
        tokenizer (transformers.BertTokenizer): An instance of BertTokenizer. Necessary to process model inputs.

    To create a BERT model for Language Model pretraining:

    .. testcode::

        from composer.models import BERTModel
        import transformers

        config = transformers.BertConfig()
        hf_model = transformers.BertLMHeadModel(config=config)
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        model = BERTModel(module=hf_model, config=config, tokenizer=tokenizer)
    """

    def __init__(self,
                 module: transformers.BertModel,
                 config: transformers.BertConfig,
                 tokenizer: Optional[transformers.BertTokenizer] = None) -> None:

        if tokenizer is None:
            model_inputs = {"input_ids", "attention_mask", "token_type_ids"}
        else:
            model_inputs = set(tokenizer.model_input_names)

        super().__init__(
            module=module,  #type: ignore (thirdparty)
            config=config,
            model_inputs=model_inputs)

        # we're going to remove the label from the expected inputs
        # since we will handle metric calculation with TorchMetrics instead of HuggingFace.
        self.model_inputs.remove("labels")

        # When using Evaluators, the validation metrics represent all possible
        # validation metrics that can be used with the bert model
        # The Evaluator class checks if it's metrics are in the models validation metrics

        ignore_index = -100
        self.val_metrics = [
            Accuracy(),
            MeanSquaredError(),
            SpearmanCorrCoef(),
            BinaryF1Score(),
            MatthewsCorrCoef(num_classes=config.num_labels),
            LanguageCrossEntropy(ignore_index=ignore_index, vocab_size=config.num_labels),
            MaskedAccuracy(ignore_index=ignore_index),
        ]
        self.train_metrics = []

    def loss(self, outputs: Mapping, batch: Batch) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')

    def validate(self, batch: BatchDict) -> BatchPair:
        """Runs the validation step.

        Args:
            batch (BatchDict): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in :meth:`.ComposerTransformer.get_model_inputs`.

        Returns:
            tuple (Tensor, Tensor): with the output from the forward pass and the correct labels.
                This is fed into directly into the output of :meth:`.ComposerModel.metrics`.
        """
        assert self.training is False, "For validation, model must be in eval mode"

        # temporary hack until eval on multiple datasets is finished
        labels = batch.pop('labels')
        output = self.forward(batch)
        output = output['logits']

        # if we are in the single class case, then remove the classes dimension
        if output.shape[1] == 1:
            output = output.squeeze(dim=1)

        return output, labels

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        return MetricCollection(self.train_metrics) if train else MetricCollection(self.val_metrics)
