# Copyright 2021 MosaicML. All Rights Reserved.

"""The ComposerModel base interface."""
from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from composer.core.types import Batch
from composer.models.nlp_metrics import LanguageCrossEntropyLoss

if TYPE_CHECKING:
    import transformers

log = logging.getLogger(__name__)

__all__ = ["ComposerModel", "ComposerTransformer"]


class ComposerModel(torch.nn.Module, abc.ABC):
    """The interface needed to make a PyTorch model compatible with :class:`composer.Trainer`.

    To create a :class:`.Trainer`\\-compatible model, subclass :class:`.ComposerModel` and
    implement :meth:`forward` and :meth:`loss`. For full functionality (logging and validation), implement :meth:`metrics`
    and :meth:`validate`.

    See the :doc:`Composer Model walk through </composer_model>` for more details.

    Minimal Example:

    .. code-block:: python

        import torchvision
        import torch.nn.functional as F

        from composer.models import ComposerModel

        class ResNet18(ComposerModel):

            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18() # define PyTorch model in __init__.

            def forward(self, batch): # batch is the output of the dataloader
                # specify how batches are passed through the model
                inputs, _ = batch
                return self.model(inputs)

            def loss(self, outputs, batch):
                # pass batches and `forward` outputs to the loss
                _, targets = batch
                return F.cross_entropy(outputs, targets)
    """

    @abc.abstractmethod
    def forward(self, batch: Batch) -> Union[Tensor, Sequence[Tensor]]:
        """Compute model output given a batch from the dataloader.

        Args:
            batch (~composer.core.types.Batch): The output batch from dataloader.

        Returns:
            Tensor | Sequence[Tensor]:
                The result that is passed to :meth:`loss` as the parameter :attr:`outputs`.

        .. warning:: This method is different from vanilla PyTorch ``model.forward(x)`` or ``model(x)`` as it takes a
                     batch of data that has to be unpacked.

        Example:

        .. code-block:: python

            def forward(self, batch): # batch is the output of the dataloader
                inputs, _ = batch
                return self.model(inputs)

        The outputs of :meth:`forward` are passed to :meth:`loss` by the trainer:

        .. code-block:: python

            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model.forward(batch)
                loss = model.loss(outputs, batch)
                loss.backward()
        """
        pass

    @abc.abstractmethod
    def loss(self, outputs: Any, batch: Batch, *args, **kwargs) -> Union[Tensor, Sequence[Tensor]]:
        """Compute the loss of the model given ``outputs`` from :meth:`forward` and a
        :class:`~composer.core.types.Batch` of data from the dataloader. The :class:`.Trainer`
        will call ``.backward()`` on the returned loss.

        Args:
            outputs (Any): The output of the forward pass.
            batch (~composer.core.types.Batch): The output batch from dataloader.

        Returns:
            Tensor | Sequence[Tensor]: The loss as a :class:`torch.Tensor`.

        Example:

        .. code-block:: python

            import torch.nn.functional as F

            def loss(self, outputs, batch):
                # pass batches and :meth:`forward` outputs to the loss
                 _, targets = batch # discard inputs from batch
                return F.cross_entropy(outputs, targets)

        The outputs of :meth:`forward` are passed to :meth:`loss` by the trainer:

        .. code-block:: python

            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model.forward(batch)
                loss = model.loss(outputs, batch)
                loss.backward()
        """
        pass

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        """Get metrics for evaluating the model. Metrics should be instances of :class:`torchmetrics.Metric` defined in
        :meth:`__init__`. This format enables accurate distributed logging. Metrics consume the outputs of
        :meth:`validate`. To track multiple metrics, return a list of metrics in a :ref:`MetricCollection
        </pages/overview.rst#metriccollection>`.

        Args:
            train (bool, optional): True to return metrics that should be computed
                during training and False otherwise. This flag is set automatically by the
                :class:`.Trainer`. Default: ``False``.

        Returns:
             Metric or MetricCollection: An instance of :class:`~torchmetrics.Metric` or :ref:`MetricCollection </pages/overview.rst#metriccollection>`.

        .. warning:: Each metric keeps states which are updated with data seen so far.
                     As a result, different metric instances should be used for training
                     and validation. See:
                     https://torchmetrics.readthedocs.io/en/latest/pages/overview.html
                     for more details.

        Example:

        .. code-block:: python

            from torchmetrics.classification import Accuracy
            from composer.models.loss import CrossEntropyLoss

            def __init__(self):
                super().__init__()
                self.train_acc = Accuracy() # torchmetric
                self.val_acc = Accuracy()
                self.val_loss = CrossEntropyLoss()

            def metrics(self, train: bool = False):
                return self.train_acc if train else MetricCollection([self.val_acc, self.val_loss])
        """
        raise NotImplementedError('Implement metrics in your ComposerModel to run validation.')

    def validate(self, batch: Batch) -> Tuple[Any, Any]:
        """Compute model outputs on provided data. Will be called by the trainer with :class:`torch.no_grad` enabled.

        The output of this function will be directly used as input
        to all metrics returned by :meth:`metrics`.

        Args:
            batch (~composer.core.types.Batch): The output batch from dataloader

        Returns:
            Tuple[Any, Any]: A Tuple of (``outputs``, ``targets``) that is passed directly to the
                :meth:`~torchmetrics.Metric.update` methods of the metrics returned by :meth:`metrics`.

        Example:

        .. code-block:: python

            def validate(self, batch): # batch is the output of the dataloader
                inputs, targets = batch
                outputs = self.model(inputs)
                return outputs, targets # return a tuple of (outputs, targets)


        This pseudocode illustrates how :meth:`validate` outputs are passed to :meth:`metrics`:

        .. code-block:: python

            metrics = model.metrics(train=False) # get torchmetrics

            for batch in val_dataloader:
                outputs, targets = model.validate(batch)
                metrics.update(outputs, targets)  # update metrics with output, targets for each batch

            metrics.compute() # compute final metrics
        """
        raise NotImplementedError('Implement validate in your ComposerModel to run validation.')


class ComposerTransformer(ComposerModel):
    """The ComposerModel base interface for Transformers.

    Works with `Hugging Face Transformers <https://huggingface.co/transformers/>`_.

    Args:
        module (transformers.PreTrainedModel): An instance of PreTrainedModel that
            contains the forward pass function.
        config (transformers.PretrainedConfig): The PretrainedConfig object that
            stores information about the model hyperparameters.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for this model,
            necessary to assert required model inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
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
        self.model_inputs.update({"labels"})

        # define metrics for measurements
        self.train_loss = LanguageCrossEntropyLoss()
        self.val_loss = LanguageCrossEntropyLoss()

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
