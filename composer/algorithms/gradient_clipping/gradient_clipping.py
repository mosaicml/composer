# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core gradient clipping classes and functions."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Union

import torch
from torch.distributed.fsdp import FullyShardedDataParallel

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.models import ComposerModel

log = logging.getLogger(__name__)

__all__ = ['GradientClipping', 'apply_gradient_clipping']


def apply_gradient_clipping(
    model: Union[ComposerModel, torch.nn.Module],
    clipping_type: str,
    clipping_threshold: float,
    fsdp_enabled: bool,
):
    """Clips all gradients in model based on specified clipping_type.

    Args:
        model (ComposerModel or torch.nn.Module): The model that we want to apply gradient clipping.
        clipping_type ('adaptive', 'norm', 'value'): String denoting which type of
            gradient clipping to do. The options are: 'norm', which clips the gradient norm
            and uses `torch.nn.utils.clip_grad_norm_`, 'value', which clips gradient at
            a specified value and uses `torch.nn.utils.clip_grad_value_`, and 'adaptive',
            which clips all gradients based on gradient norm:parameter norm ratio using
            composer.algorithms.gradient_clipping.gradient_clipping._apply_agc.
        clipping_threshold (float, optional): Specifies what value to clip the gradients
            to (for 'value'), what values to clip the gradient norms to (for 'norm'), and
            threshold by which if grad_norm / weight_norm is greater than this threshold then
            scale gradients by this threshold * (weight_norm / grad_norm) (for 'adaptive').
        fsdp_enabled (bool): Bool of if the model is a FSDP model or not.
    """
    if fsdp_enabled:
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel) and module.check_is_root():
                if clipping_type == 'norm':
                    module.clip_grad_norm_(max_norm=clipping_threshold)
                elif clipping_type == 'value':
                    module.clip_grad_norm_(max_norm=clipping_threshold, norm_type=float('inf'))
                else:
                    raise ValueError(f"clipping type must be 'norm' or 'value' with FSDP not {clipping_type}")
    else:
        parameters = model.parameters()
        if clipping_type == 'adaptive':
            _apply_agc(parameters, clipping_threshold=clipping_threshold)
        elif clipping_type == 'norm':
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=clipping_threshold)
        elif clipping_type == 'value':
            torch.nn.utils.clip_grad_value_(parameters, clip_value=clipping_threshold)
        else:
            raise ValueError(f"clipping_type must be 'adaptive', 'norm', or 'value' not {clipping_type} ")


def _apply_agc(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    clipping_threshold: float,
) -> None:
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.
    Args:
        parameters (torch.Tensor or Iterable[torch.Tensor]): The parameters to of the
            model for whose gradients we will clip
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.
    """
    for param in parameters:
        if param.grad is None:
            continue

        # Detach weights and gradients, so the clipping operation is not added to
        # computational graph.
        weights = param.detach()
        grad = param.grad.detach()

        # Get clipped version of gradients.
        clipped_grad_coeff = _get_clipped_gradient_coeff(weights, grad, clipping_threshold=clipping_threshold)

        # Copy clipped gradients into param.grad attribute, so they can be accessed by
        # optimizer.
        grad.mul_(clipped_grad_coeff)


class GradientClipping(Algorithm):
    """Clips all gradients in model based on specified clipping_type.

    Runs on ``Event.AFTER_TRAIN_BATCH``.

    Example:
         .. testcode::

            from composer.algorithms import GradientClipping
            from composer.trainer import Trainer
            gc = GradientClipping(clipping_type='norm', clipping_threshold=0.1)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[gc],
                optimizers=[optimizer]
            )

    Args:
        clipping_type ('adaptive', 'norm', 'value'): String denoting which type of
            gradient clipping to do. The options are: 'norm', which clips the gradient norm
            and uses `torch.nn.utils.clip_grad_norm_`, 'value', which clips gradient at
            a specified value and uses `torch.nn.utils.clip_grad_value_`, and 'adaptive',
            which clips all gradients based on gradient norm:parameter norm ratio using
            composer.algorithms.gradient_clipping.gradient_clipping._apply_agc.
        clipping_threshold (float, optional): Specifies what value to clip the gradients
            to (for 'value'), what values to clip the gradient norms to (for 'norm'), and
            threshold by which if grad_norm / weight_norm is greater than this threshold then
            scale gradients by this threshold * (weight_norm / grad_norm) (for 'adaptive').
    """

    def __init__(self, clipping_type: str, clipping_threshold: float):
        self.clipping_type = clipping_type
        self.clipping_threshold = clipping_threshold

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.INIT, Event.AFTER_TRAIN_BATCH]

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        if event == Event.AFTER_TRAIN_BATCH:
            apply_gradient_clipping(
                model=state.model,
                clipping_type=self.clipping_type,
                clipping_threshold=self.clipping_threshold,
                fsdp_enabled=state.fsdp_enabled,
            )


def _get_clipped_gradient_coeff(weights: torch.Tensor, grad: torch.Tensor, clipping_threshold: float = 0.01):
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.

    Gradients whose norms exceed

    .. math:: weight_norm * clipping_threshold

    are scaled down by

    .. math:: (weight_norm / grad_norm) * clipping_threshold.

    Args:
        weights (torch.Tensor): Tensor of weights (parameters) from the model.
        grad (torch.Tensor): Tensor of gradients of the loss with respect to the weights.
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.

    Return:
        clipped_grad_coeff (torch.Tensor): Coefficient of same shape as grad_norm equal to
            (weight_norm / grad_norm) * clipping_threshold for gradients whose norms
            that exceed weight_norm * clipping_threshold and one otherwise.
    """

    # Compute and clamp grad and weight norms.
    w_norm = _unitwise_norm(weights)
    grad_norm = _unitwise_norm(grad)

    # Gradients whose norms are greater than weight_norm * clipping_threhsold are
    # scaled down by (weight_norm * clipping_threhsold) / grad_norm.
    max_norm = w_norm.mul_(clipping_threshold)
    clipped_grad_coeff = max_norm.div_(grad_norm).nan_to_num_(nan=1.0).clamp_(max=1.0)

    return clipped_grad_coeff


def _unitwise_norm(tensor: torch.Tensor):
    """Implements unitwise norm as described in Brock et al, 2021.

    For 0D scalars of shape [], we trivially normalize with dim=0 which essentially returns the absolute value of the scalar.
    For 1D *.bias weights of shape [out_features], we normalize across entire vector -> dim=0.
    For 2D torch.nn.Linear weights of shape [out_features, in_features]: we normalize across in_features -> dim = 1
    For 4D torch.nn.Conv2d weights [out_channels, in_channels, kernel_height, kernel_width]:
        we normalize across [in_channels, kernel_height, kernel_width] -> dim = (1, 2, 3).
    If a 3D parameter were somehow in your model, we would normalize buy the last two dimensions -> dim = (1,2).

    Args:
        tensor (torch.Tensor): A parameter or gradient of the model.

    Returns:
        The appropriate L2 norm of the parameter or gradient as described above.
    """
    # 0D for scalars, 1D for bias vectors.
    if tensor.ndim <= 1:
        dim = 0
        keepdim = False
    # 2D corresponds to MLPs and 4D corresponds to ConvNets.
    else:
        dim = tuple(range(1, tensor.ndim))
        keepdim = True
    # L2 Norm.
    return torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
