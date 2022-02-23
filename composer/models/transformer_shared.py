# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Tuple

from composer.models.base import ComposerModel
from composer.models.nlp_metrics import LanguageCrossEntropyLoss

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch, Metrics, Tensors

log = logging.getLogger(__name__)


class ComposerTransformer(ComposerModel):
    """Implements the base logic that all Transformers can build on top of.

    Works with `Hugging Face Transformers <https://huggingface.co/transformers/>`_.

    Args:
        module (transformers.PreTrainedModel): An instance of PreTrainedModel that
            contains the forward pass function.
        config (transformers.PretrainedConfig): The PretrainedConfig object that
            stores information about the model hyperparameters.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for this model,
            necessary to assert required model inputs.
    """

    def __init__(self,
                 module: transformers.PreTrainedModel,
                 config: transformers.PretrainedConfig,
                 tokenizer: transformers.PreTrainedTokenizer,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.tokenizer = tokenizer
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

        if gradient_checkpointing:
            self.module.gradient_checkpointing_enable()  # type: ignore

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        """Computes the loss of the tensor from the output.

        We don't implement this for the generic Transformer abstraction, since loss
        functions are model and objective specific. A single model architecture could
        use a myriad of loss functions which are better left expressed by the user.

        Args:
            outputs (Mapping): The dictionary output from the model.
                It could contain the loss as computed by Hugging Face,
                or algorithms can pop the labels from the input in case
                they modify the loss function.
            batch (Batch): The set of ground truth labels to use to compute the loss against.

        Returns:
            The loss as a ``Tensors`` object.

        Raises:
            NotImplementedError: A model-specific and task-specific loss function must be written.
        """

        raise NotImplementedError("A model-specific loss function must be written.")

    def forward(self, batch: Batch) -> Mapping:
        """Runs the forward pass of the model.

        Args:
            batch (Batch): A dictionary of Dict[str, Tensor] of inputs that the
                model expects, as found in ComposerTransformer.get_model_inputs().

        Returns:
            A dictionary of model outputs as a ``Mapping``. It will include the loss
            if `labels` is passed as an input.
        """
        if not isinstance(batch, dict):
            raise ValueError(f'Model expects batch to be a dict, got {type(batch)}')

        for key in self.model_inputs:
            if key not in batch.keys():
                raise ValueError(f'Batch missing key: {key}')

        output = self.module(**batch)  # type: ignore (thirdparty)
        return output

    def metrics(self, train: bool = False) -> Metrics:
        """Get metrics for evaluating the model.

        Downstream models should override this method if they would like to
        add task-specific metrics.

        Args:
            train (bool): a boolean flag to indicate whether to return
                training or validation metrics.

        .. warning:: If train=True, then it might calculate the training loss twice if
                     algorithms are overriding the loss fn. This could be expensive due
                     to the computational cost of softmax; it is worth exploring caching strategies.

        Returns:
            A  Metrics object that can be used to calculate task performance.
        """
        return self.train_loss if train else self.val_loss

    def validate(self, batch: Batch) -> Tuple[Mapping, None]:
        """Runs the validation step.

        Args:
            batch (Batch): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in ComposerTransformer.get_model_inputs().

        Returns:
            Tuple[Mapping, None]: A tuple containing the output from the forward pass.
                This is fed into directly into the output of :meth:`metrics`.
        """

        assert self.training is False, "For validation, model must be in eval mode"
        output = self.forward(batch)

        return (output, None)

    def get_model_inputs(self):
        """Returns a set of inputs that the model expects in the forward pass.

        If an algorithm wants to interact with the model inputs (for instance,
        popping the labels for a custom loss fn, or adding attention head masks
        for head pruning, it must access self.set_model_inputs().

        Returns:
            The set of keys that are expected in the Mapping used to compute the forward pass.
        """

        return self.model_inputs
