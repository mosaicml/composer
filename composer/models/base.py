# Copyright 2021 MosaicML. All Rights Reserved.

"""The ComposerModel base interface."""
from __future__ import annotations

import abc
from typing import Any, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from composer.core.types import Batch

__all__ = ["ComposerModel"]


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
