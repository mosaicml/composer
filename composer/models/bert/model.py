from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

import transformers
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.collections import MetricCollection

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
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

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

        return (output, labels)

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection([self.train_loss, self.train_acc]) if train else MetricCollection(
            [self.val_loss, self.val_acc])
