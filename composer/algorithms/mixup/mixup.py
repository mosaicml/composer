# Copyright 2021 MosaicML. All Rights Reserved.

"""Core MixUp classes and functions."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.models.loss import ensure_targets_one_hot

log = logging.getLogger(__name__)

__all__ = ["MixUp", "mixup_batch"]


def mixup_batch(input: torch.Tensor,
                target: torch.Tensor,
                mixing: Optional[float] = None,
                alpha: float = 0.2,
                indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create new samples using convex combinations of pairs of samples.

    This is done by taking a convex combination of ``input`` with a randomly
    permuted copy of ``input``. The permutation takes place along the sample
    axis (dim 0).

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
        indices (Tensor, optional): Permutation of the samples to use.
            Default: ``None``.

    Returns:
        input_mixed (torch.Tensor): batch of inputs after mixup has been applied
        target_orig (torch.Tensor): The original labels
        target_perm (torch.Tensor): The labels of the mixed in examples
        mixing (torch.Tensor): the amount of mixing used.

    Example:
        .. testcode::

            import torch
            from composer.functional import mixup_batch

            N, C, H, W = 2, 3, 4, 5
            X = torch.randn(N, C, H, W)
            y = torch.randint(num_classes, size=(N,))
            X_mixed, y_orig, y_perm, mixing = mixup_batch(
                X, y, alpha=0.2)
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

    return x_mixup, target, permuted_target, mixing


class MixUp(Algorithm):
    """`MixUp <https://arxiv.org/abs/1710.09412>`_ trains the network on convex combinations of pairs of examples and
    targets rather than individual examples and targets.

    This is done by taking a convex combination of a given batch X with a
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
            A useful trick when using a cross entropy loss. Default: ``True``

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

    def __init__(self, alpha: float = 0.2, interpolate_loss: bool = True):
        self.alpha = alpha
        self.interpolate_loss = interpolate_loss
        self.mixing = 0.0
        self.indices = torch.Tensor()
        self.permuted_target = torch.Tensor()

    def match(self, event: Event, state: State) -> bool:
        if self.interpolate_loss:
            return event in [Event.AFTER_DATALOADER, Event.AFTER_LOSS]
        else:
            return event in [Event.AFTER_DATALOADER, Event.BEFORE_LOSS]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        input, target = state.batch_pair

        if event == Event.AFTER_DATALOADER:
            input, target = state.batch_pair
            assert isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor), \
                "Multiple tensors for inputs or targets not supported yet."

            self.mixing = _gen_mixing_coef(self.alpha)
            self.indices = _gen_indices(input.shape[0])

            new_input, _, self.permuted_target, _ = mixup_batch(
                input,
                target,
                mixing=self.mixing,
                indices=self.indices,
            )

            state.batch = (new_input, target)

        if self.interpolate_loss and event == Event.AFTER_LOSS:
            assert isinstance(state.loss, torch.Tensor), "Multiple losses not supported yet"
            # Interpolate the loss
            modified_batch = (input, self.permuted_target)
            if isinstance(state.model, DistributedDataParallel):
                loss_fn = state.model.module.loss
            elif isinstance(state.model, ComposerModel):
                loss_fn = state.model.loss
            else:
                raise RuntimeError("Model must be of type ComposerModel or DistributedDataParallel")

            def loss(outputs, batch):
                return loss_fn(outputs, batch)

            new_loss = loss(state.outputs, modified_batch)
            state.loss *= (1 - self.mixing)
            state.loss += self.mixing * new_loss

        if not self.interpolate_loss and event == Event.BEFORE_LOSS:
            # Interpolate the targets
            input, target = state.batch_pair
            assert isinstance(state.outputs, torch.Tensor), "Multiple output tensors not supported yet"
            assert isinstance(target, torch.Tensor), "Multiple target tensors not supported yet"
            # Make sure that the targets are dense/one-hot
            target = ensure_targets_one_hot(state.outputs, target)
            permuted_target = ensure_targets_one_hot(state.outputs, self.permuted_target)
            # Interpolate to get the new target
            mixed_up_target = (1 - self.mixing) * target + self.mixing * permuted_target
            # Create the new batch
            state.batch = (input, mixed_up_target)


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
