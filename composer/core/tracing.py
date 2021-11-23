# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Any, Optional, Tuple

import torch
from torchmetrics.collections import MetricCollection

from composer.core.types import Batch, Metrics, Tensors
from composer.trainer.trainer import BaseMosaicModel
from composer.utils.device_helpers import move_batch_to_gpu


def trace_mosaic_model(model: BaseMosaicModel,
                       example_input: Batch,
                       device: str = "cpu",
                       save_file_path: Optional[str] = None):
    if not device in ("cpu", "gpu"):
        raise ValueError(f"Invalid device {device} must be either 'cpu' or 'gpu'.")

    if device == "gpu":
        # just use one GPU for tracing
        torch_device = torch.device(f"cuda:0")
        model.to(torch_device)
        example_input = move_batch_to_gpu(batch=example_input, device=torch_device)

    output = model.forward(batch=example_input)

    jit_model = torch.jit.trace_module(model, {'forward': (example_input,), 'loss': (output, example_input)})

    if save_file_path:
        torch.jit.save(jit_model, save_file_path)

    return jit_model


def load_model_trace(filename: str) -> BaseMosaicModel:
    jit_model = torch.jit.load(filename)

    model = _EmptyMosaicModel()
    model.forward = jit_model.forward  # type: ignore
    model.loss = jit_model.loss  # type: ignore
    return model


class _EmptyMosaicModel(BaseMosaicModel):
    """An empty implementation of :class:`~composer.trainer.trainer.BaseMosaicModel`.

    This class contains stub definitions for all the required methods in
    :class:`~composer.trainer.trainer.BaseMosaicModel`. The intended use is for
    loading models that have been traced as type BaseMosaicModel.
    See :meth:`~composer.utils.load_model_trace` for an example.
    """

    def loss(self, outputs: Any, batch: Batch, *args, **kwargs) -> Tensors:
        """Compute the loss of the model.

        Args:
            outputs (Any): The output of the forward pass.
            batch (~composer.core.types.Batch): The input batch from dataloader.

        Returns:
            Tensors:
                A stub of the loss as a ``Tensors`` object.
        """
        del outputs, batch, args, kwargs
        return torch.ones((1,))

    def forward(self, batch: Batch) -> Tensors:
        """Compute model output given an input.

        Args:
            batch (Batch): The input batch for the forward pass.

        Returns:
            Tensors:
                The result that is passed to :meth:`loss` as a ``Tensors``
                object.
        """
        del batch
        return torch.ones((1,))

    def metrics(self, train: bool = False) -> Metrics:
        """Get metrics for evaluating the model.

        .. warning:: Each metric keeps states which are updated with data seen so far.
                     As a result, different metric instances should be used for training
                     and validation. See:
                     https://torchmetrics.readthedocs.io/en/latest/pages/overview.html
                     for more details.

        Args:
            train (bool, optional): True to return metrics that should be computed
                during training and False otherwise. (default: ``False``)

        Returns:
            Metrics: A ``Metrics`` object.
        """
        del train
        return MetricCollection([])

    def validate(self, batch: Batch) -> Tuple[Any, Any]:
        """Compute model outputs on provided data.

        The output of this function will be directly used as input
        to all metrics returned by :meth:`metrics`.

        Args:
            batch (Batch): The data to perform validation with.
                Specified as a tuple of tensors (input, target).

        Returns:
            Tuple[Any, Any]: Tuple that is passed directly to the
                `update()` methods of the metrics returned by :meth:`metrics`.
                Most often, this will be a tuple of the form (predictions, targets).
        """
        del batch
        return (1, 1)
