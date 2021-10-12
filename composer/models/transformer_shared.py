# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Tuple

import transformers

from composer.models.base import BaseMosaicModel
from composer.models.nlp_metrics import LanguageCrossEntropyLoss

if TYPE_CHECKING:
    from composer.core.types import Batch, Metrics, Tensors

log = logging.getLogger(__name__)


class MosaicTransformer(BaseMosaicModel):
    """
    Implements the base logic that all Transformers can build on top off.

    Args:
        module (transformers.PreTrainedModel): An instance of PreTrainedModel that contains the forward pass function.
        config (transformers.PretrainedConfig): The PretrainedConfig object that stores information about the model
                                                hyperparameters.
        tokenizer_name (str): The name of the tokenizer used for tihs model, necessary to assert required model inputs.
    """

    def __init__(self, module: transformers.PreTrainedModel, config: transformers.PretrainedConfig,
                 tokenizer_name: str) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        log.info("Number of parameters in the model: " \
                 f"{sum(p.numel() for p in module.parameters()):,}")  # type: ignore (thirdparty)
        log.info("Number of trainable parameters in the model: "
                 f"{sum(p.numel() for p in module.parameters() if p.requires_grad):,}")  # type: ignore (thirdparty)

        # the set of inputs that a model expects
        # if an algorithm modifies the loss function, it must remove "labels" from this set.
        self.model_inputs = set(self.tokenizer.model_input_names)
        self.model_inputs.update(set({"labels"}))

        # define metrics for measurements
        self.train_loss = LanguageCrossEntropyLoss()
        self.val_loss = LanguageCrossEntropyLoss()

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        """
        Computes the loss of the tensor from the output.

        Args:
            outputs (Mapping): The dictionary output from the model. It could contain the loss as computed by HF, or
                               algorithms can pop the labels from the input in case they modify the loss function.
            batch (Batch): the set of ground truth labels to use to compute the loss against.

        Returns:
            None

        Raises:
            NotImplementedError: a model-specific and task-specific loss function must be written.

        Notes:
            We don't implement this for the generic Transformer abstraction, since loss functions are model and
            objective specific. A single model architeture could use a myriad of loss functions which are better left
            epressed by the user.
        """

        raise NotImplementedError("A model-specific loss function must be written.")

    def forward(self, batch: Batch) -> Mapping:
        """
        Runs the forward pass of the model.

        Args:
            batch (Batch): a dictionary of Dict[str, Tensor] of inputs that the model expects, as found in
            MosaicTransformer.get_model_inputs().

        Returns:
            output (Mapping): a dictionary of model outputs. It will include the loss if `labels` is passed as an input.
        """
        if not isinstance(batch, dict):
            raise ValueError(f'Model expects batch to be a dict, got {type(batch)}')

        for key in self.model_inputs:
            if key not in batch.keys():
                raise ValueError(f'Batch missing key: {key}')

        output = self.module(**batch)  # type: ignore (thirdparty)
        return output

    def metrics(self, train: bool = False) -> Metrics:
        """
        Returns the Metrics objects for computing the training and validation losses.
        Downstream models should override this method if they would like to add task-specific metrics.


        Args:
            train (bool): a boolean flag to indicate whether to return training or validation metrics.

        Returns:
            A  Metrics object that can be used to calculate task performance.

        Notes:
            If train=True, then it might calculate the training loss twice if algorithms are overriding the loss fn.
            This could be expensive due to the computational cost of softmax; it is worth exploring caching stratgies.
        """
        return self.train_loss if train else self.val_loss

    def validate(self, batch: Batch) -> Tuple[Mapping, None]:
        """
        Runs the validation step.

        Args:
            batch (Batch): a dictionary of Dict[str, Tensor] of inputs that the model expects, as found in
            MosaicTransformer.get_model_inputs().

        Returns:
            A tuple of (Mapping, None) with the output from the forward pass. This is fed into a Metrics object.
        """

        assert self.training is False, "For validation, model must be in eval mode"
        output = self.forward(batch)

        return (output, None)

    def get_model_inputs(self):
        """
        Returns a set of inputs that the model expects in the forward pass.

        Args:
            None

        Returns:
            self.modellinputs (set): the set of keys that are expected in the Mapping used to compute the forward pass.

        Notes:
            If an algorithm wants to interact with the model inputs (for instance, popping the labels for a custom loss
            fn, or adding attention head masks for head pruning, it must access self.set_model_inputs().
        """

        return self.model_inputs
