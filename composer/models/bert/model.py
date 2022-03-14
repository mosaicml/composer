# Copyright 2021 MosaicML. All Rights Reserved.

"""Implements a BERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

from torchmetrics import Accuracy, MatthewsCorrcoef, MeanSquaredError, SpearmanCorrcoef
from torchmetrics.collections import MetricCollection

from composer.models.nlp_metrics import BinaryF1Score, CrossEntropyLoss, MaskedAccuracy
from composer.models.transformer_shared import ComposerTransformer

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch, BatchDict, Metrics, Tensors

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

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')

    def validate(self, batch: BatchDict) -> Tuple[Tensors, Tensors]:
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

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection(self.train_metrics) if train else MetricCollection(self.val_metrics)
