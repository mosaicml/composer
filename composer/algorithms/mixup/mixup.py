# Copyright 2021 MosaicML. All Rights Reserved.

"""Core MixUp classes and functions."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from composer.core.types import Algorithm, Event, Logger, State, Tensor
from composer.models.loss import check_for_index_targets

log = logging.getLogger(__name__)

__all__ = ["MixUp", "mixup_batch"]


def mixup_batch(x: Tensor,
                y: Tensor,
                n_classes: int,
                interpolation_lambda: Optional[float] = None,
                alpha: float = 0.2,
                indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create new samples using convex combinations of pairs of samples.

    This is done by taking a convex combination of x with a randomly
    permuted copy of x. The interpolation parameter lambda should be chosen from
    a ``Beta(alpha, alpha)`` distribution for some parameter alpha > 0.
    Note that the same lambda is used for all examples within the batch.

    Both the original and shuffled labels are returned. This is done because
    for many loss functions (such as cross entropy) the targets are given as
    indices, so interpolation must be handled separately.

    Example:
         .. testcode::

            from composer.algorithms.mixup import mixup_batch
            new_inputs, new_targets, perm = mixup_batch(
                                                x=X_example,
                                                y=y_example,
                                                n_classes=1000,
                                                alpha=0.2
                                                )

    Args:
        x: input tensor of shape (B, d1, d2, ..., dn), B is batch size, d1-dn
            are feature dimensions.
        y: target tensor of shape (B, f1, f2, ..., fm), B is batch size, f1-fn
            are possible target dimensions.
        interpolation_lambda: coefficient used to interpolate between the
            two examples. If provided, must be in ``[0, 1]``. If ``None``,
            value is drawn from a ``Beta(alpha, alpha)`` distribution.
        alpha: parameter for the beta distribution over the
            ``interpolation_lambda``. Only used if ``interpolation_lambda``
            is not provided.
        n_classes: total number of classes.
        indices: Permutation of the batch indices `1..B`. Used
            for permuting without randomness.

    Returns:
        x_mix: batch of inputs after mixup has been applied
        y_mix: labels after mixup has been applied
        perm: the permutation used
    """
    if interpolation_lambda is None:
        interpolation_lambda = _gen_interpolation_lambda(alpha)
    # Create shuffled versions of x and y in preparation for interpolation
    # Use given indices if there are any.
    if indices is None:
        shuffled_idx = torch.randperm(x.shape[0])
    else:
        shuffled_idx = indices
    x_shuffled = x[shuffled_idx]
    y_shuffled = y[shuffled_idx]
    # Interpolate between the inputs
    x_mix = (1 - interpolation_lambda) * x + interpolation_lambda * x_shuffled

    # First check if labels are indices. If so, convert them to onehots.
    # This is under the assumption that the loss expects torch.LongTensor, which is true for pytorch cross_entropy
    if check_for_index_targets(y):
        y_onehot = F.one_hot(y, num_classes=n_classes)
        y_shuffled_onehot = F.one_hot(y_shuffled, num_classes=n_classes)
        y_mix = ((1. - interpolation_lambda) * y_onehot + interpolation_lambda * y_shuffled_onehot)
    else:
        y_mix = ((1. - interpolation_lambda) * y + interpolation_lambda * y_shuffled)
    return x_mix, y_mix, shuffled_idx


class MixUp(Algorithm):
    """`MixUp <https://arxiv.org/abs/1710.09412>`_ trains the network on convex combinations of pairs of examples and
    targets rather than individual examples and targets.

    This is done by taking a convex combination of a given batch X with a
    randomly permuted copy of X. The mixing coefficient is drawn from a
    Beta(``alpha``, ``alpha``) distribution.

    Training in this fashion sometimes reduces generalization error.

    Example:
         .. testcode::

            from composer.algorithms import MixUp
            from composer.trainer import Trainer
            mixup_algorithm = MixUp(num_classes=1000, alpha=0.2)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[mixup_algorithm],
                optimizers=[optimizer]
            )

    Args:
        num_classes (int): the number of classes in the task labels.
        alpha (float): the psuedocount for the Beta distribution used to sample
            interpolation parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair.
    """

    def __init__(self, num_classes: int, alpha: float = 0.2):
        self.num_classes = num_classes
        self.alpha = alpha
        self._interpolation_lambda = 0.0
        self._indices = torch.Tensor()

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.INIT and Event.AFTER_DATALOADER.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        return event == Event.AFTER_DATALOADER

    @property
    def interpolation_lambda(self) -> float:
        return self._interpolation_lambda

    @interpolation_lambda.setter
    def interpolation_lambda(self, new_int_lamb: float) -> None:
        self._interpolation_lambda = new_int_lamb

    @property
    def indices(self) -> Tensor:
        return self._indices

    @indices.setter
    def indices(self, new_indices: Tensor) -> None:
        self._indices = new_indices

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

        self.interpolation_lambda = _gen_interpolation_lambda(self.alpha)

        new_input, new_target, self.indices = mixup_batch(
            x=input,
            y=target,
            interpolation_lambda=self.interpolation_lambda,
            n_classes=self.num_classes,
        )

        state.batch = (new_input, new_target)


def _gen_interpolation_lambda(alpha: float) -> float:
    """Generates ``Beta(alpha, alpha)`` distribution."""
    # First check if alpha is positive.
    assert alpha >= 0
    # Draw the interpolation parameter from a beta distribution.
    # Check here is needed because beta distribution requires alpha > 0
    # but alpha = 0 is fine for mixup.
    if alpha == 0:
        interpolation_lambda = 0
    else:
        interpolation_lambda = np.random.beta(alpha, alpha)
    # for symmetric beta distribution, can always use 0 <= lambda <= .5;
    # this way the "main" label is always the original one, which keeps
    # the training accuracy meaningful
    return max(interpolation_lambda, 1. - interpolation_lambda)
