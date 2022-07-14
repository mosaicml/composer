# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts 🤗 Transformers models to composer models"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from torchmetrics import Metric
from torchmetrics.collections import MetricCollection

from composer.models.base import ComposerModel
from composer.utils.import_helpers import MissingConditionalImportError

if TYPE_CHECKING:
    import transformers

__all__ = ['HuggingFaceModel']


class HuggingFaceModel(ComposerModel):
    """
    A wrapper class that converts 🤗 Transformers models to composer models.

    Args:
        model (transformers.PreTrainedModel): A 🤗 Transformers model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to prepare the dataset and validate model inputs during training. Default ``None``.
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
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

        super().__init__()
        self.model = model
        self.config = model.config

        # the set of inputs that a model expects inferred from the model type or
        # tokenizer if provided
        if tokenizer is None:
            if isinstance(self.model.base_model, transformers.GPT2Model):
                self.model_inputs = {'input_ids', 'attention_mask'}
            elif isinstance(self.model.base_model, transformers.BertModel):
                self.model_inputs = {'input_ids', 'attention_mask', 'token_type_ids'}
        else:
            assert tokenizer.model_input_names is not None, 'the tokenizer should have a model input name'
            self.model_inputs = set(tokenizer.model_input_names)

        self.use_logits = use_logits

        self.train_metrics = None
        self.valid_metrics = None

        if metrics:
            metric_collection = MetricCollection(metrics)
            self.train_metrics = metric_collection.clone(prefix='train_')
            self.valid_metrics = metric_collection.clone(prefix='val_')

    def forward(self, batch):
        for key in self.model_inputs:
            if key not in batch.keys():
                raise ValueError(f'Batch missing key: {key}')

        output = self.model(**batch)  # type: ignore (thirdparty)
        return output

    def loss(self, outputs, batch):
        return outputs['loss']

    def validate(self, batch):
        if self.use_logits:
            labels = batch.pop('labels')
            output = self.forward(batch)
            output = output['logits']

            # if we are in the single class case, then remove the classes dimension
            if output.shape[1] == 1:
                output = output.squeeze(dim=1)

            return output, labels
        else:
            output = self.forward(batch)
            return output, None

    def metrics(self, train: bool = False):
        return self.train_metrics if train else self.valid_metrics

    def get_model_inputs(self):
        """Returns a set of inputs that the model expects in the forward pass.
        If an algorithm wants to interact with the model inputs (for instance,
        popping the labels for a custom loss fn, or adding attention head masks
        for head pruning, it must access self.set_model_inputs().
        Returns:
            model_inputs: The set of keys that are expected in the Mapping used to compute the forward pass.
        """

        return self.model_inputs
