# Copyright 2021 MosaicML. All Rights Reserved.

"""Core MixUp classes and functions."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from composer.core.types import Algorithm, Event, Logger, State, Tensor
from composer.models.loss import _check_for_index_targets

log = logging.getLogger(__name__)

__all__ = ["MixUp", "mixup_batch"]


def mixup_batch(input: Tensor,
                target: Tensor,
                num_classes: int,
                mixing: Optional[float] = None,
                alpha: float = 0.2,
                indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        num_classes (int): total number of classes or output variables
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
        target_mixed (torch.Tensor): labels after mixup has been applied
        perm (torch.Tensor): the permutation used

    Example:
        .. testcode::

            import torch
            from composer.functional import mixup_batch

            N, C, H, W = 2, 3, 4, 5
            num_classes = 10
            X = torch.randn(N, C, H, W)
            y = torch.randint(num_classes, size=(N,))
            X_mixed, y_mixed, perm = mixup_batch(
                X, y, num_classes=num_classes, alpha=0.2)
    """
    if mixing is None:
        mixing = _gen_mixing_coef(alpha)
    # Create shuffled versions of x and y in preparation for interpolation
    # Use given indices if there are any.
    if indices is None:
        shuffled_idx = torch.randperm(input.shape[0])
    else:
        shuffled_idx = indices
    x_shuffled = input[shuffled_idx]
    y_shuffled = target[shuffled_idx]
    # Interpolate between the inputs
    x_mix = (1 - mixing) * input + mixing * x_shuffled

    # First check if labels are indices. If so, convert them to onehots.
    # This is under the assumption that the loss expects torch.LongTensor, which is true for pytorch cross_entropy
    if _check_for_index_targets(target):
        y_onehot = F.one_hot(target, num_classes=num_classes)
        y_shuffled_onehot = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mix = ((1. - mixing) * y_onehot + mixing * y_shuffled_onehot)
    else:
        y_mix = ((1. - mixing) * target + mixing * y_shuffled)
    return x_mix, y_mix, shuffled_idx


class MixUp(Algorithm):
    """`MixUp <https://arxiv.org/abs/1710.09412>`_ trains the network on convex combinations of pairs of examples and
    targets rather than individual examples and targets.

    This is done by taking a convex combination of a given batch X with a
    randomly permuted copy of X. The mixing coefficient is drawn from a
    ``Beta(alpha, alpha)`` distribution.

    Training in this fashion sometimes reduces generalization error.

    Args:
        num_classes (int): the number of classes in the task labels.
        alpha (float, optional): the psuedocount for the Beta distribution used to sample
            mixing parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair. Default: ``0.2``.

    Example:
        .. testcode::

            from composer.algorithms import MixUp
            algorithm = MixUp(num_classes=10, alpha=0.2)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(self, num_classes: int, alpha: float = 0.2):
        self.num_classes = num_classes
        self.alpha = alpha
        self.mixing = 0.0
        self.indices = torch.Tensor()

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.INIT and Event.AFTER_DATALOADER.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Applies MixUp augmentation on State input.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """

        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors for inputs or targets not supported yet."

        self.mixing = _gen_mixing_coef(self.alpha)

        new_input, new_target, self.indices = mixup_batch(
            input,
            target,
            mixing=self.mixing,
            num_classes=self.num_classes,
        )

        state.batch = (new_input, new_target)


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
    return max(mixing_lambda, 1. - mixing_lambda)
