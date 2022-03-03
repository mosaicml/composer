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
                 num_classes: int,
                 cut_proportion: Optional[float] = None,
                 alpha: float = 1.,
                 bbox: Optional[Tuple] = None,
                 indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create new samples using combinations of pairs of samples.

    This is done by masking a region of ``X`` and filling the masked region with
    the corresponding content in a permuted copy ``X``. The permutation takes
    place along the sample axis (dim 0), so that each output image has part
    of it replaced with content from another image.

    The area of the masked region is determined by ``cut_proportion``, which
    must be in :math:`(0, 1)` if provided. If not provided, ``cut_proportion``
    is drawn from a :math:`Beta(alpha, alpha)` distribution for some parameter
    ``alpha > 0``. The original paper used a fixed value of ``alpha = 1``.

    Note that the same ``cut_proportion`` and masked region are used for
    the whole batch.

    Args:
        X (torch.Tensor): input tensor of shape ``(N, C, H, W)``
        y (torch.Tensor): target tensor of either shape ``N`` or
            ``(N, num_classes)``. In the former case, elements of ``y`` must
            be integer class ids in the range ``0..num_classes``. In the
            latter case, rows of ``y`` may be arbitrary vectors of targets,
            including, e.g., one-hot encoded class labels, smoothed class
            labels, or multi-output regression targets.
        num_classes (int): total number of classes or output variables
        cut_proportion (float, optional): relative area of cutmix region
            compared to the original size. Must be in the interval
            :math:`(0, 1)`. If ``None``, value is drawn from a
            ``Beta(alpha, alpha)`` distribution.
        alpha (float, optional): parameter for the Beta distribution over
            ``cut_proportion``. Ignored if ``cut_proportion`` is provided.
        bbox (Tuple, optional): predetermined ``(rx1, ry1, rx2, ry2)``
            coordinates of the bounding box.
        indices (torch.Tensor, optional): Permutation of the samples to use.

    Returns:
        X_mixed: batch of inputs after cutmix has been applied.
        y_mixed: soft labels for mixed input samples. These are a convex
            combination of the (possibly one-hot-encoded) labels from the
            original samples and the samples chosen to fill the masked
            regions, with the relative weighting equal to ``cut_proportion``.
            E.g., if a sample was originally an image with label ``0`` and
            40% of the image of was replaced with data from an image with label
            ``2``, the resulting labels, assuming only three classes, would be
            ``[1, 0, 0] * 0.6 + [0, 0, 1] * 0.4 = [0.6, 0, 0.4]``.
        perm: the permutation used

    Example:
        .. testcode::

            import torch
            from composer.functional import cutmix_batch

            N, C, H, W = 2, 3, 4, 5
            num_classes = 10
            X = torch.randn(N, C, H, W)
            y = torch.randint(num_classes, size=(N,))
            X_mixed, y_mixed = cutmix_batch(
                X, y, num_classes=num_classes, alpha=0.2)

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
    if cut_proportion is None:
        cut_proportion = _gen_cutmix_coef(alpha)
    if bbox:
        rx, ry, rw, rh = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        rx, ry, rw, rh = _rand_bbox(X.shape[2], X.shape[3], cut_proportion)
        bbox = (rx, ry, rw, rh)

    # Fill in the box with a part of a random image.
    X_cutmix[:, :, rx:rw, ry:rh] = X_cutmix[shuffled_idx, :, rx:rw, ry:rh]
    # adjust lambda to exactly match pixel ratio. This is an implementation detail taken from
    # the original implementation, and implies lambda is not actually beta distributed.
    adjusted_lambda = _adjust_lambda(cut_proportion, X, bbox)

    # Make a shuffled version of y for interpolation
    y_shuffled = y[shuffled_idx]
    # Interpolate between labels using the adjusted lambda
    # First check if labels are indices. If so, convert them to onehots.
    # This is under the assumption that the loss expects torch.LongTensor, which is true for pytorch cross_entropy
    if check_for_index_targets(y):
        y_onehot = F.one_hot(y, num_classes=num_classes)
        y_shuffled_onehot = F.one_hot(y_shuffled, num_classes=num_classes)
        y_cutmix = adjusted_lambda * y_onehot + (1 - adjusted_lambda) * y_shuffled_onehot
    else:
        y_cutmix = adjusted_lambda * y + (1 - adjusted_lambda) * y_shuffled

    return X_cutmix, y_cutmix, shuffled_idx


class CutMix(Algorithm):
    """`CutMix <https://arxiv.org/abs/1905.04899>`_ trains the network on non-overlapping combinations of pairs of
    examples and iterpolated targets rather than individual examples and targets.

    This is done by taking a non-overlapping combination of a given batch X with a
    randomly permuted copy of X. The area is drawn from a :math:`Beta(\alpha, \alpha)`
    distribution.

    Training in this fashion sometimes reduces generalization error.

    Args:
        num_classes (int): the number of classes in the task labels.
        alpha (float): the psuedocount for the Beta distribution used to sample
            area parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair.

    Example:
        .. testsetup::

            import torch
            from composer import models
            from composer.algorithms import CutMix
            from composer.trainer import Trainer

            # create dataloaders and optimizer
            num_batches, batch_size, num_features = 2, 3, 5
            num_classes = 10
            X_train = torch.randn(num_batches, num_features)
            y_train = torch.randint(num_classes, size=(num_batches, batch_size))
            X_val = torch.randn(num_batches, num_features)
            y_val = torch.randint(num_classes, size=(num_batches, batch_size))
            train_dataloader = torch.utils.data.DataLoader(zip(X_train, y_train))
            eval_dataloader = torch.utils.data.DataLoader(zip(X_val, y_val))

            # create model and optimizer
            model = models.MNIST_Classifier(num_classes=num_classes)
            optimizer = torch.optim.Adam(model.parameters())


        .. testcode::

            algorithm = CutMix(num_classes=num_classes, alpha=0.2)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(self, num_classes: int, alpha: float = 1.):
        self.num_classes = num_classes
        self.alpha = alpha
        self._indices = torch.Tensor()
        self._cutmix_lambda = 0.0
        self._bbox: Tuple[int, int, int, int] = 0, 0, 0, 0

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

        # these are saved only for testing
        self._indices = _gen_indices(input)
        self._cutmix_lambda = _gen_cutmix_coef(alpha)
        self._bbox = _rand_bbox(input.shape[2], input.shape[3], self._cutmix_lambda)
        self._cutmix_lambda = _adjust_lambda(self._cutmix_lambda, input, self._bbox)

        new_input, new_target, _ = cutmix_batch(
            X=input,
            y=target,
            num_classes=self.num_classes,
            alpha=alpha,
            cut_proportion=self._cutmix_lambda,
            bbox=self._bbox,
            indices=self._indices,
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


def _gen_cutmix_coef(alpha: float) -> float:
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
