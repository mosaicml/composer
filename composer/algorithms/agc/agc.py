# Copyright 2021 MosaicML. All Rights Reserved.

"""Core automatic gradient clipping classes and functions."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ["AGC", "apply_agc"]


def apply_agc(
    model: torch.nn.Module,
    clipping_threshold: float = 0.01,
    eps: float = 1e-3,
) -> None:
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.

    Example:
         .. testcode::

            from composer.algorithms.agc import apply_agc
            apply_agc(model=model)


    Args:
        model (torch.nn.Module): The model being trained.
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.
        eps (float, optional): Minimum value that weight norms are clamped to.
    """
    for param in model.parameters():
        if param.grad is None:
            continue

        # Detach weights and gradients, so the clipping operation is not added to
        # computational graph.
        weights = param.detach()
        grad = param.grad.detach()

        # Get clipped version of gradients.
        clipped_grad_coeff = _get_clipped_gradient_coeff(weights, grad)

        # Copy clipped gradients into param.grad attribute, so they can be accessed by
        # optimizer.
        grad.mul_(clipped_grad_coeff)


class AGC(Algorithm):
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.

    From <https://arxiv.org/abs/2102.06171>.
    Computes the norm of the weights and the norm of their corresponding gradients, then
    scales the gradients by (weight_norm / grad_norm) * clipping_threshold for gradients
    whose norms are greater than weight_norm * clipping_threshold. Norms are taken across
    rows for weight matrices in MLPs, across entire filters/kernels for CNNs (channel and
    spatial dimensions), and across the whole vector for biases.

    Runs on ``Event.AFTER_TRAIN_BATCH``.

    Example:
         .. testcode::

            from composer.algorithms import AGC
            from composer.trainer import Trainer
            agc_algorithm = AGC()
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[agc_algorithm],
                optimizers=[optimizer]
            )

    Args:
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.
        eps (float, optional): Minimum value that weight norms are clamped to.
    """

    def __init__(self, clipping_threshold: float = 0.01, eps: float = 1e-3):
        self.clipping_threshold = clipping_threshold
        self.eps = eps

    def match(self, event: Event, state: State) -> bool:
        """Run on ``Event.AFTER_TRAIN_BATCH``."""
        return event == Event.AFTER_TRAIN_BATCH

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Freeze layers in the model."""
        apply_agc(model=state.model, clipping_threshold=self.clipping_threshold, eps=self.eps)


# Factored this to a function to enable easier testing.
def _get_clipped_gradient_coeff(weights: torch.Tensor,
                                grad: torch.Tensor,
                                clipping_threshold: float = 0.01,
                                eps: float = 1e-3):
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.

    Gradients whose norms exceed weight_norm * clipping_threshold are scaled down by
    (weight_norm / grad_norm) * clipping_threshold.

    Args:
        weights (torch.Tensor): Tensor of weights (parameters) from the model.
        grad (torch.Tensor): Tensor of gradients of the loss with respect to the weights.
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.
        eps (float, optional): Minimum value that weight norms are clamped to.

    Return:
        clipped_grad_coeff (torch.Tensor): Coefficient of same shape as grad_norm equal to
            (weight_norm / grad_norm) * clipping_threshold for gradients whose norms
            that exceed weight_norm * clipping_threshold and one otherwise.
    """

    # Compute and clamp grad and weight norms.
    w_norm = _unitwise_norm(weights).clamp_(min=eps)
    grad_norm = _unitwise_norm(grad).clamp_(min=1e-6)

    # Gradients whose norms are greater than weight_norm * clipping_threhsold are
    # scaled down by (weight_norm * clipping_threhsold) / grad_norm.
    max_norm = w_norm * clipping_threshold
    clipped_grad_coeff = torch.where(grad_norm > max_norm, max_norm / grad_norm, torch.ones_like(grad_norm))

    return clipped_grad_coeff


def _unitwise_norm(tensor: torch.Tensor):
    """Implements unitwise norm as described in Brock et al, 2021.

    For bias vectors (1D Tensors): normalize across entire vector.
    For MLP Weight matrix (2D tensors): we normalize across rows (dim = 1)
    For CNNs (4D Tensors): we normalzie across the entire kernel (channel, height,
         and width) -> dim = (1, 2, 3).

    Args:
        tensor (torch.Tensor): A parameter or gradient of the model.

    Returns:
        The appropriate L2 norm of the parameter or gradient (norm of rows for 2D,
        norm of 3D kernels for 4D tensor (last three dims)).
    """
    if tensor.ndim <= 1:
        dim = None
        keepdim = False
    # 2D corresponds to MLPs and 4D corresponds to ConvNets.
    else:
        dim = tuple(range(1, tensor.ndim))
        keepdim = True
    # L2 Norm.
    return torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
