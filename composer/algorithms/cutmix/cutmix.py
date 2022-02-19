# Copyright 2021 MosaicML. All Rights Reserved.

"""Core CutMix classes and functions."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from composer.core.types import Algorithm, Event, Logger, State, Tensor
from composer.models.loss import check_for_index_targets

log = logging.getLogger(__name__)

__all__ = ["CutMix", "cutmix_batch"]


def cutmix_batch(X: Tensor,
                 y: Tensor,
                 n_classes: int,
                 alpha: float = 1.,
                 cutmix_lambda: Optional[float] = None,
                 bbox: Optional[Tuple] = None,
                 indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create new samples using combinations of pairs of samples.

    This is done by masking a region of X, and filling the masked region with a
    permuted copy of x. The cutmix parameter lambda should be chosen from
    a Beta(alpha, alpha) distribution for some parameter alpha > 0. The area of
    the masked region is determined by lambda, and so labels are interpolated accordingly.
    Note that the same lambda is used for all examples within the batch. The original
    paper used a fixed value of alpha = 1.

    Both the original and shuffled labels are returned. This is done because
    for many loss functions (such as cross entropy) the targets are given as
    indices, so interpolation must be handled separately.

    Example:
         .. testcode::

            from composer.algorithms.cutmix import cutmix_batch
            new_input_batch = cutmix_batch(
                X=X_example,
                y=y_example,
                n_classes=1000,
                alpha=1.0
            )

    Args:
        X: input tensor of shape (B, d1, d2, ..., dn), B is batch size, d1-dn
            are feature dimensions.
        y: target tensor of shape (B, f1, f2, ..., fm), B is batch size, f1-fn
            are possible target dimensions.
        n_classes: total number of classes.
        alpha: parameter for the beta distribution of the cutmix region size.
        cutmix_lambda: optional, fixed size of cutmix region.
        bbox: optional, predetermined (rx1, ry1, rx2, ry2) coords of the bounding box.
        indices: Permutation of the batch indices `1..B`. Used
            for permuting without randomness.

    Returns:
        X_cutmix: batch of inputs after cutmix has been applied.
        y_cutmix: labels after cutmix has been applied.
    """
    # Create shuffled indicies across the batch in preparation for cutting and mixing.
    # Use given indices if there are any.
    if indices is None:
        shuffled_idx = _gen_indices(X)
    else:
        shuffled_idx = indices

    # Create the new inputs.
    X_cutmix = torch.clone(X)
    # Sample a rectangular box using lambda. Use variable names from the paper.
    if cutmix_lambda is None:
        cutmix_lambda = _gen_cutmix_lambda(alpha)
    if bbox:
        rx, ry, rw, rh = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        rx, ry, rw, rh = _rand_bbox(X.shape[2], X.shape[3], cutmix_lambda)
        bbox = (rx, ry, rw, rh)

    # Fill in the box with a part of a random image.
    X_cutmix[:, :, rx:rw, ry:rh] = X_cutmix[shuffled_idx, :, rx:rw, ry:rh]
    # adjust lambda to exactly match pixel ratio. This is an implementation detail taken from
    # the original implementation, and implies lambda is not actually beta distributed.
    adjusted_lambda = _adjust_lambda(cutmix_lambda, X, bbox)

    # Make a shuffled version of y for interpolation
    y_shuffled = y[shuffled_idx]
    # Interpolate between labels using the adjusted lambda
    # First check if labels are indices. If so, convert them to onehots.
    # This is under the assumption that the loss expects torch.LongTensor, which is true for pytorch cross_entropy
    if check_for_index_targets(y):
        y_onehot = F.one_hot(y, num_classes=n_classes)
        y_shuffled_onehot = F.one_hot(y_shuffled, num_classes=n_classes)
        y_cutmix = adjusted_lambda * y_onehot + (1 - adjusted_lambda) * y_shuffled_onehot
    else:
        y_cutmix = adjusted_lambda * y + (1 - adjusted_lambda) * y_shuffled

    return X_cutmix, y_cutmix


class CutMix(Algorithm):
    """`CutMix <https://arxiv.org/abs/1905.04899>`_ trains the network on non-overlapping combinations of pairs of
    examples and iterpolated targets rather than individual examples and targets.

    This is done by taking a non-overlapping combination of a given batch X with a
    randomly permuted copy of X. The area is drawn from a ``Beta(alpha, alpha)``
    distribution.

    Training in this fashion sometimes reduces generalization error.

    Example:
         .. testcode::

            from composer.algorithms import CutMix
            from composer.trainer import Trainer
            cutmix_algorithm = CutMix(num_classes=1000, alpha=1.0)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[cutmix_algorithm],
                optimizers=[optimizer]
            )

    Args:
        num_classes (int): the number of classes in the task labels.
        alpha (float): the psuedocount for the Beta distribution used to sample
            area parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair.
    """

    def __init__(self, num_classes: int, alpha: float = 1.):
        self.num_classes = num_classes
        self.alpha = alpha
        self._indices = torch.Tensor()
        self._cutmix_lambda = 0.0
        self._bbox = tuple()

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
    def indices(self) -> Tensor:
        return self._indices

    @indices.setter
    def indices(self, new_indices: Tensor) -> None:
        self._indices = new_indices

    @property
    def cutmix_lambda(self) -> float:
        return self._cutmix_lambda

    @cutmix_lambda.setter
    def cutmix_lambda(self, new_lambda: float) -> None:
        self._cutmix_lambda = new_lambda

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self._bbox

    @bbox.setter
    def bbox(self, new_bbox: Tuple[int, int, int, int]) -> None:
        self._bbox = new_bbox

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Applies CutMix augmentation on State input.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """

        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors for inputs or targets not supported yet."
        alpha = self.alpha

        self.indices = _gen_indices(input)
        self.cutmix_lambda = _gen_cutmix_lambda(alpha)
        self.bbox = _rand_bbox(input.shape[2], input.shape[3], self.cutmix_lambda)
        self.cutmix_lambda = _adjust_lambda(self.cutmix_lambda, input, self.bbox)

        new_input, new_target = cutmix_batch(
            X=input,
            y=target,
            n_classes=self.num_classes,
            alpha=alpha,
            cutmix_lambda=self.cutmix_lambda,
            bbox=self.bbox,
            indices=self.indices,
        )

        state.batch = (new_input, new_target)


def _gen_indices(x: Tensor) -> Tensor:
    """Generates indices of a random permutation of elements of a batch.

    Args:
        x: input tensor of shape (B, d1, d2, ..., dn), B is batch size, d1-dn
            are feature dimensions.

    Returns:
        indices: A random permutation of the batch indices.
    """
    return torch.randperm(x.shape[0])


def _gen_cutmix_lambda(alpha: float) -> float:
    """Generates lambda from ``Beta(alpha, alpha)``

    Args:
        alpha: Parameter for the Beta(alpha, alpha) distribution

    Returns:
        cutmix_lambda: Lambda parameter for performing cutmix.
    """
    # First check if alpha is positive.
    assert alpha >= 0
    # Draw the area parameter from a beta distribution.
    # Check here is needed because beta distribution requires alpha > 0
    # but alpha = 0 is fine for cutmix.
    if alpha == 0:
        cutmix_lambda = 0
    else:
        cutmix_lambda = np.random.beta(alpha, alpha)
    return cutmix_lambda


def _rand_bbox(W: int,
               H: int,
               cutmix_lambda: float,
               cx: Optional[int] = None,
               cy: Optional[int] = None) -> Tuple[int, int, int, int]:
    """Randomly samples a bounding box with area determined by cutmix_lambda.

    Adapted from original implementation https://github.com/clovaai/CutMix-PyTorch

    Args:
        W: Width of the image
        H: Height of the image
        cutmix_lambda: Lambda param from cutmix, used to set the area of the box.
        cx: Optional x coordinate of the center of the box.
        cy: Optional y coordinate of the center of the box.

    Returns:
        bbx1: Leftmost edge of the bounding box
        bby1: Top edge of the bounding box
        bbx2: Rightmost edge of the bounding box
        bby2: Bottom edge of the bounding box
    """
    cut_ratio = np.sqrt(1.0 - cutmix_lambda)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # uniform
    if cx is None:
        cx = np.random.randint(W)
    if cy is None:
        cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def _adjust_lambda(cutmix_lambda: float, x: Tensor, bbox: Tuple) -> float:
    """Rescale the cutmix lambda according to the size of the clipped bounding box.

    Args:
        cutmix_lambda: Lambda param from cutmix, used to set the area of the box.
        x: input tensor of shape (B, d1, d2, ..., dn), B is batch size, d1-dn
            are feature dimensions.
        bbox: (x1, y1, x2, y2) coordinates of the boundind box, obeying x2 > x1, y2 > y1.

    Returns:
        adjusted_lambda: Rescaled cutmix_lambda to account for part of the bounding box
            being potentially out of bounds of the input.
    """
    rx, ry, rw, rh = bbox[0], bbox[1], bbox[2], bbox[3]
    adjusted_lambda = 1 - ((rw - rx) * (rh - ry) / (x.size()[-1] * x.size()[-2]))
    return adjusted_lambda
