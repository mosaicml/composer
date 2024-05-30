# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core MixUp classes and functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.loss.utils import ensure_targets_one_hot

log = logging.getLogger(__name__)

__all__ = ['MixUp', 'mixup_batch']


def mixup_batch(
    input: torch.Tensor,
    target: torch.Tensor,
    mixing: Optional[float] = None,
    alpha: float = 0.2,
    indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Create new samples using convex combinations of pairs of samples.

    This is done by taking a convex combination of ``input`` with a randomly
    permuted copy of ``input``. The permutation takes place along the sample
    axis (``dim=0``).

    The relative weight of the original ``input`` versus the permuted copy is
    defined by the ``mixing`` parameter. This parameter should be chosen
    from a ``Beta(alpha, alpha)`` distribution for some parameter ``alpha > 0``.
    Note that the same ``mixing`` is used for the whole batch.

    Args:
        input (torch.Tensor): input tensor of shape ``(minibatch, ...)``, where
            ``...`` indicates zero or more dimensions.
        target (torch.Tensor): target tensor of shape ``(minibatch, ...)``, where
            ``...`` indicates zero or more dimensions.
        mixing (float, optional): coefficient used to interpolate
            between the two examples. If provided, must be in :math:`[0, 1]`.
            If ``None``, value is drawn from a ``Beta(alpha, alpha)``
            distribution. Default: ``None``.
        alpha (float, optional): parameter for the Beta distribution over
            ``mixing``. Ignored if ``mixing`` is provided. Default: ``0.2``.
        indices (torch.Tensor, optional): Permutation of the samples to use.
            Default: ``None``.

    Returns:
        input_mixed (torch.Tensor): batch of inputs after mixup has been applied
        target_perm (torch.Tensor): The labels of the mixed-in examples
        mixing (torch.Tensor): the amount of mixing used

    Example:
        .. testcode::

            import torch
            from composer.functional import mixup_batch

            N, C, H, W = 2, 3, 4, 5
            X = torch.randn(N, C, H, W)
            y = torch.randint(num_classes, size=(N,))
            X_mixed, y_perm, mixing = mixup_batch(
                X,
                y,
                alpha=0.2,
            )
    """
    if mixing is None:
        mixing = _gen_mixing_coef(alpha)
    # Create permuted versions of x and y in preparation for interpolation
    # Use given indices if there are any.
    if indices is None:
        permuted_idx = _gen_indices(input.shape[0])
    else:
        permuted_idx = indices
    x_permuted = input[permuted_idx]
    permuted_target = target[permuted_idx]
    # Interpolate between the inputs
    x_mixup = (1 - mixing) * input + mixing * x_permuted

    return x_mixup, permuted_target, mixing


class MixUp(Algorithm):
    """`MixUp <https://arxiv.org/abs/1710.09412>`_ trains the network on convex batch combinations.


    The algorithm uses individual examples and targets to make a convex combination of a given batch X with a
    randomly permuted copy of X. The mixing coefficient is drawn from a
    ``Beta(alpha, alpha)`` distribution.

    Training in this fashion sometimes reduces generalization error.

    Args:
        alpha (float, optional): the psuedocount for the Beta distribution used to sample
            mixing parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair. Default: ``0.2``.
        interpolate_loss (bool, optional): Interpolates the loss rather than the labels.
            A useful trick when using a cross entropy loss. Will produce incorrect behavior
            if the loss is not a linear function of the targets. Default: ``False``
        input_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
        target_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.

    Example:
        .. testcode::

            from composer.algorithms import MixUp
            algorithm = MixUp(alpha=0.2)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(
        self,
        alpha: float = 0.2,
        interpolate_loss: bool = False,
        input_key: Union[str, int, tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, tuple[Callable, Callable], Any] = 1,
    ):
        self.alpha = alpha
        self.interpolate_loss = interpolate_loss
        self.mixing = 0.0
        self.indices = torch.Tensor()
        self.permuted_target = torch.Tensor()
        self.input_key, self.target_key = input_key, target_key

    def match(self, event: Event, state: State) -> bool:
        if self.interpolate_loss:
            return event in [Event.BEFORE_FORWARD, Event.BEFORE_BACKWARD]
        else:
            return event in [Event.BEFORE_FORWARD, Event.BEFORE_LOSS]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        input, target = state.batch_get_item(key=self.input_key), state.batch_get_item(key=self.target_key)

        if event == Event.BEFORE_FORWARD:
            if not isinstance(input, torch.Tensor):
                raise NotImplementedError('Multiple tensors for inputs not supported yet.')
            if not isinstance(target, torch.Tensor):
                raise NotImplementedError('Multiple tensors for targets not supported yet.')

            self.mixing = _gen_mixing_coef(self.alpha)
            self.indices = _gen_indices(input.shape[0])

            new_input, self.permuted_target, _ = mixup_batch(
                input,
                target,
                mixing=self.mixing,
                indices=self.indices,
            )

            state.batch_set_item(self.input_key, new_input)

        if not self.interpolate_loss and event == Event.BEFORE_LOSS:
            # Interpolate the targets
            if not isinstance(state.outputs, torch.Tensor):
                raise NotImplementedError('Multiple output tensors not supported yet')
            if not isinstance(target, torch.Tensor):
                raise NotImplementedError('Multiple target tensors not supported yet')
            # Make sure that the targets are dense/one-hot
            target = ensure_targets_one_hot(state.outputs, target)
            permuted_target = ensure_targets_one_hot(state.outputs, self.permuted_target)
            # Interpolate to get the new target
            mixed_up_target = (1 - self.mixing) * target + self.mixing * permuted_target
            # Create the new batch
            state.batch_set_item(self.target_key, mixed_up_target)

        if self.interpolate_loss and event == Event.BEFORE_BACKWARD:
            # Grab the loss function
            if hasattr(state.model, 'loss'):
                loss_fn = state.model.loss
            elif hasattr(state.model, 'module') and hasattr(state.model.module, 'loss'):
                if isinstance(state.model.module, torch.nn.Module):
                    loss_fn = state.model.module.loss
                else:
                    raise TypeError('state.model.module must be a torch module')
            else:
                raise AttributeError('Loss must be accesable via model.loss or model.module.loss')
            # Verify that the loss is callable
            if not callable(loss_fn):
                raise TypeError('Loss must be callable')
            # Interpolate the loss
            new_loss = loss_fn(state.outputs, (input, self.permuted_target))
            if not isinstance(state.loss, torch.Tensor):
                raise NotImplementedError('Multiple losses not supported yet')
            if not isinstance(new_loss, torch.Tensor):
                raise NotImplementedError('Multiple losses not supported yet')
            state.loss = (1 - self.mixing) * state.loss + self.mixing * new_loss


def _gen_mixing_coef(alpha: float) -> float:
    """Samples ``max(z, 1-z), z ~ Beta(alpha, alpha)``."""
    # First check if alpha is positive.
    assert alpha >= 0
    # Draw the mixing parameter from a beta distribution.
    # Check here is needed because beta distribution requires alpha > 0
    # but alpha = 0 is fine for mixup.
    if alpha == 0:
        mixing_lambda = 0
    else:
        mixing_lambda = np.random.beta(alpha, alpha)
    # for symmetric beta distribution, can always use 0 <= lambda <= .5;
    # this way the "main" label is always the original one, which keeps
    # the training accuracy meaningful
    return min(mixing_lambda, 1. - mixing_lambda)


def _gen_indices(num_samples: int) -> torch.Tensor:
    """Generates a random permutation of the batch indices."""
    return torch.randperm(num_samples)
