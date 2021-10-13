# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import yahp as hp
from torch.nn import functional as F

from composer.algorithms import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State, Tensor

log = logging.getLogger(__name__)


def gen_interpolation_lambda(alpha: float) -> float:
    """Generates ``Beta(alpha, alpha)`` distribution"""
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


def mixup_batch(x: Tensor,
                y: Tensor,
                interpolation_lambda: float,
                n_classes: int,
                indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create new samples using convex combinations of pairs of samples.

    This is done by taking a convex combination of x with a randomly
    permuted copy of x. The interploation parameter lambda should be chosen from
    a ``Beta(alpha, alpha)`` distribution for some parameter alpha > 0.
    Note that the same lambda is used for all examples within the batch.

    Both the original and shuffled labels are returned. This is done because
    for many loss functions (such as cross entropy) the targets are given as
    indices, so interpolation must be handled separately.

    Args:
        x: input tensor of shape (B, d1, d2, ..., dn), B is batch size, d1-dn
            are feature dimensions.
        y: target tensor of shape (B, f1, f2, ..., fm), B is batch size, f1-fn
            are possible target dimensions.
        interpolation_lambda: amount of interpolation based on alpha.
        n_classes: total number of classes.
        indices: Permutation of the batch indices `1..B`. Used
            for permuting without randomness.

    Returns:
        x_mix: batch of inputs after mixup has been applied
        y_mix: labels after mixup has been applied
        perm: the permutation used

    Example:
        from composer import functional as CF

        for X, y in dataloader:
            l = CF.gen_interpolation_lambda(alpha=0.2)
            X, y, _ = CF.mixup_batch(X, y, l, nclasses)

            pred = model(X)
            loss = loss_fun(pred, y)  # loss_fun must accept dense labels (ie NOT indices)

    """
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

    y_onehot = F.one_hot(y, num_classes=n_classes)
    y_shuffled_onehot = F.one_hot(y_shuffled, num_classes=n_classes)
    y_mix = ((1. - interpolation_lambda) * y_onehot + interpolation_lambda * y_shuffled_onehot)

    return x_mix, y_mix, shuffled_idx


@dataclass
class MixUpHparams(AlgorithmHparams):
    """See :class:`MixUp`"""

    alpha: float = hp.required('Strength of interpolation, should be >= 0. No interpolation if alpha=0.',
                               template_default=0.2)

    def initialize_object(self) -> "MixUp":
        return MixUp(**asdict(self))


class MixUp(Algorithm):
    """`MixUp <https://arxiv.org/abs/1710.09412>`_ trains the network on
    convex combinations of pairs of examples and targets rather than individual
    examples and targets.

    This is done by taking a convex combination of a given batch X with a
    randomly permuted copy of X. The mixing coefficient is drawn from a
    ``Beta(alpha, alpha)`` distribution.

    Training in this fashion reduces generalization error.

    Args:
        alpha: the psuedocount for the Beta distribution used to sample
            interpolation parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair.
    """

    def __init__(self, alpha: float):
        self.hparams = MixUpHparams(alpha=alpha)

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.INIT and Event.AFTER_DATALOADER

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        return event in (Event.AFTER_DATALOADER, Event.INIT)

    @property
    def interpolation_lambda(self) -> float:
        return self._interpolation_lambda

    @interpolation_lambda.setter
    def interpolation_lambda(self, new_int_lamb) -> None:
        self._interpolation_lambda = new_int_lamb

    @property
    def indices(self) -> Tensor:
        return self._indices

    @indices.setter
    def indices(self, new_indices: Tensor) -> None:
        self._indices = new_indices

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Applies MixUp augmentation on State input

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger

        """
        if event == Event.INIT:
            self.num_classes: int = state.model.num_classes  # type: ignore
            return

        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors for inputs or targets not supported yet."
        alpha = self.hparams.alpha

        self.interpolation_lambda = gen_interpolation_lambda(alpha)

        new_input, new_target, self.indices = mixup_batch(
            x=input,
            y=target,
            interpolation_lambda=self.interpolation_lambda,
            n_classes=self.num_classes,
        )

        state.batch = (new_input, new_target)
