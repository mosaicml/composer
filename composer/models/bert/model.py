from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

import transformers
from torchmetrics import Accuracy, MatthewsCorrcoef, MeanSquaredError, SpearmanCorrcoef
from torchmetrics.collections import MetricCollection

from composer.models.nlp_metrics import BinaryF1Score
from composer.models.transformer_shared import MosaicTransformer

if TYPE_CHECKING:
    from composer.core.types import Batch, Metrics, Tensors


class BERTModel(MosaicTransformer):
    """
    Implements a BERT wrapper around a MosaicTransformer.
    """

    def __init__(self, module: transformers.BertModel, config: transformers.BertConfig, tokenizer_name: str) -> None:
        super().__init__(
            module=module,  #type: ignore (thirdparty)
            config=config,
            tokenizer_name=tokenizer_name)

        # we're going to remove the label from the expected inputs
        # since we will handle metric calculation with TorchMetrics instead of HuggingFace.
        self.model_inputs.remove("labels")

        # if config.num_labels=1, then we are training a regression task, so we should update our loss functions
        self.train_metrics = []
        self.val_metrics = []

        if config.num_labels == 1:
            self.train_loss = MeanSquaredError()
            self.val_loss = MeanSquaredError()

            self.train_spearman = SpearmanCorrcoef()
            self.val_spearman = SpearmanCorrcoef()

            self.train_metrics.extend([self.train_loss, self.train_spearman])
            self.val_metrics.extend([self.val_loss, self.val_spearman])

        if config.num_labels == 2:
            # due to how F1 is calculated in TorchMetrics, we force multiclass = False to examine binary F1.
            # TODO (Moin): make sure this is moved to be dataset-specific
            self.train_f1 = BinaryF1Score()
            self.val_f1 = BinaryF1Score()

            self.train_metrics.extend([self.train_f1])
            self.val_metrics.extend([self.val_f1])

        if config.num_labels > 1:
            self.train_acc = Accuracy()
            self.val_acc = Accuracy()

            self.train_matthews = MatthewsCorrcoef(num_classes=config.num_labels)
            self.val_matthews = MatthewsCorrcoef(num_classes=config.num_labels)

            self.train_metrics.extend([self.train_acc, self.train_matthews])
            self.val_metrics.extend([self.val_acc, self.val_matthews])

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')

    def validate(self, batch: Batch) -> Tuple[Mapping, None]:
        """Runs the validation step.

        Args:
            batch (Batch): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in MosaicTransformer.get_model_inputs().

        Returns:
            A tuple of (Mapping, None) with the output from the forward pass.
            This is fed into directly into the output of :meth:`metrics`.
        """

        assert self.training is False, "For validation, model must be in eval mode"

        # we remove the loss from the forward pass inputs so we can calculate it independently
        labels = batch.pop('labels')
        output = self.forward(batch)
        output = output['logits']

        # if we are in the single class case, then remove the dimension for downstream metrics
        if output.shape[1] == 1:
            output = output.squeeze()

        return (output, labels)

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection(self.train_metrics) if train else MetricCollection(self.val_metrics)
