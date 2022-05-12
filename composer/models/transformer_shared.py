# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The ComposerModel base interface for Transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Sequence, Tuple, Union

from torch import Tensor

from composer.metrics.nlp import HFCrossEntropy
from composer.models.base import ComposerModel

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch

log = logging.getLogger(__name__)

__all__ = ["ComposerTransformer"]


class ComposerTransformer(ComposerModel):
    """The ComposerModel base interface for Transformers.

    Works with `Hugging Face Transformers <https://huggingface.co/transformers/>`_.

    Args:
        module (transformers.PreTrainedModel): An instance of PreTrainedModel that
            contains the forward pass function.
        config (transformers.PretrainedConfig): The PretrainedConfig object that
            stores information about the model hyperparameters.
        model_inputs (set): The dictionary keys that should be required to be fed into the model's forward function.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
    """

    def __init__(self,
                 module: transformers.PreTrainedModel,
                 config: transformers.PretrainedConfig,
                 model_inputs: set,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.module = module
        self.config = config
        log.info("Number of parameters in the model: " \
                 f"{sum(p.numel() for p in module.parameters()):,}")  # type: ignore (thirdparty)
        log.info("Number of trainable parameters in the model: "
                 f"{sum(p.numel() for p in module.parameters() if p.requires_grad):,}")  # type: ignore (thirdparty)

        # the set of inputs that a model expects
        # if an algorithm modifies the loss function, it must remove "labels" from this set.
        self.model_inputs = model_inputs
        self.model_inputs.update(set({"labels"}))

        # define metrics for measurements
        self.train_loss = HFCrossEntropy()
        self.val_loss = HFCrossEntropy()

        if gradient_checkpointing:
            self.module.gradient_checkpointing_enable()  # type: ignore

    def loss(self, outputs: Mapping, batch: Batch) -> Union[Tensor, Sequence[Tensor]]:
        """Computes the loss of the tensor from the output.

        We don't implement this for the generic Transformer abstraction, since loss
        functions are model and objective specific. A single model architecture could
        use a myriad of loss functions which are better left expressed by the user.

        Args:
            outputs (Mapping): The dictionary output from the model.
                It could contain the loss as computed by Hugging Face,
                or algorithms can pop the labels from the input in case
                they modify the loss function.
            batch (:class:`~composer.core.types.Batch`): The set of ground truth labels to use to compute the loss against.

        Raises:
            NotImplementedError: A model-specific and task-specific loss function must be written.
        """
        raise NotImplementedError("A model-specific loss function must be written.")

    def forward(self, batch: Batch) -> Mapping:
        """Runs the forward pass of the model.

        Args:
            batch (~composer.core.types.Batch): A dictionary of Dict[str, Tensor] of inputs that the
                model expects, as found in :meth:`.ComposerTransformer.get_model_inputs`.

        Returns:
            output: A dictionary of model outputs as a ``Mapping``. It will include the loss if `labels` is passed as an input.
        """
        if not isinstance(batch, dict):
            raise ValueError(f'Model expects batch to be a dict, got {type(batch)}')

        for key in self.model_inputs:
            if key not in batch.keys():
                raise ValueError(f'Batch missing key: {key}')

        output = self.module(**batch)  # type: ignore (thirdparty)
        return output

    def validate(self, batch: Batch) -> Tuple[Mapping, None]:
        """Runs the validation step.

        Args:
            batch (~composer.core.types.Batch): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in :meth:`.ComposerTransformer.get_model_inputs`.

        Returns:
            Tuple[Mapping, None]: A tuple containing the output from the forward pass.
                This is fed into directly into the output of :meth:`.ComposerModel.metrics`.
        """
        assert self.training is False, "For validation, model must be in eval mode"
        output = self.forward(batch)
        return output, None

    def get_model_inputs(self):
        """Returns a set of inputs that the model expects in the forward pass.

        If an algorithm wants to interact with the model inputs (for instance,
        popping the labels for a custom loss fn, or adding attention head masks
        for head pruning, it must access self.set_model_inputs().

        Returns:
            model_inputs: The set of keys that are expected in the Mapping used to compute the forward pass.
        """

        return self.model_inputs
