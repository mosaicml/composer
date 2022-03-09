# Copyright 2021 MosaicML. All Rights Reserved.

"""Core CutMix classes and functions."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from composer.core.types import Algorithm, Event, Logger, State, Tensor
from composer.models.loss import _check_for_index_targets

log = logging.getLogger(__name__)

__all__ = ["CutMix", "cutmix_batch"]


def cutmix_batch(input: Tensor,
                 target: Tensor,
                 num_classes: int,
                 length: Optional[float] = None,
                 alpha: float = 1.,
                 bbox: Optional[Tuple] = None,
                 indices: Optional[torch.Tensor] = None,
                 uniform_sampling: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create new samples using combinations of pairs of samples.

    This is done by masking a region of each image in ``input`` and filling
    the masked region with the corresponding content from a random different
    image in``input``.

    The position of the masked region is determined by drawing a center point
    uniformly at random from all spatial positions.

    The area of the masked region is computed using either ``length`` or
    ``alpha``. If ``length`` is provided, it directly determines the size
    of the masked region. If it is not provided, the fraction of the input
    area to mask is drawn from a ``Beta(alpha, alpha)`` distribution.
    The original paper used a fixed value of ``alpha = 1``.

    Alternatively, one may provide a bounding box to mask directly, in
    which case ``alpha`` is ignored and ``length`` must not be provided.

    The same masked region is used for the whole batch.

    .. note::
        The masked region is clipped at the spatial boundaries of the inputs.
        This means that there is no padding required, but the actual region
        used may be smaller than the nominal size computed using ``length``
        or ``alpha``.

    Args:
        input (torch.Tensor): input tensor of shape ``(N, C, H, W)``
        target (torch.Tensor): target tensor of either shape ``N`` or
            ``(N, num_classes)``. In the former case, elements of ``target``
            must be integer class ids in the range ``0..num_classes``. In the
            latter case, rows of ``target`` may be arbitrary vectors of targets,
            including, e.g., one-hot encoded class labels, smoothed class
            labels, or multi-output regression targets.
        num_classes (int): total number of classes or output variables
        length (float, optional): Relative side length of the masked region.
            If specified, ``length`` is interpreted as a fraction of ``H`` and
            ``W``, and the resulting box is of size ``(length * H, length * W)``.
            Default: ``None``.
        alpha (float, optional): parameter for the Beta distribution over
            the fraction of the input to mask. Ignored if ``length`` is
            provided. Default: ``1``.
        bbox (tuple, optional): predetermined ``(x1, y1, x2, y2)``
            coordinates of the bounding box. Default: ``None``.
        indices (torch.Tensor, optional): Permutation of the samples to use.
            Default: ``None``.
        uniform_sampling (bool, optional): If ``True``, sample the bounding box
            such that each pixel has an equal probability of being mixed.
            If ``False``, defaults to the sampling used in the original paper
            implementation. Default: ``False``.

    Returns:
        input_mixed (torch.Tensor): batch of inputs after cutmix has been
            applied.
        target_mixed (torch.Tensor): soft labels for mixed input samples.
            These are a convex combination of the (possibly one-hot-encoded)
            labels from the original samples and the samples chosen to fill
            the masked regions, with the relative weighting equal to the
            fraction of the spatial size that is cut.
            E.g., if a sample was originally an image with label ``0`` and
            40% of the image of was replaced with data from an image with label
            ``2``, the resulting labels, assuming only three classes, would be
            ``[1, 0, 0] * 0.6 + [0, 0, 1] * 0.4 = [0.6, 0, 0.4]``.

    Raises:
        ValueError: If both ``length`` and ``bbox`` are provided.

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
    if bbox is not None and length is not None:
        raise ValueError(f"Cannot provide both length and bbox; got {length} and {bbox}")

    # Create shuffled indicies across the batch in preparation for cutting and mixing.
    # Use given indices if there are any.
    if indices is None:
        shuffled_idx = _gen_indices(input)
    else:
        shuffled_idx = indices

    H, W = input.shape[-2], input.shape[-1]

    # figure out fraction of area to cut
    if length is None:
        cutmix_lambda = _gen_cutmix_coef(alpha)
    else:
        cut_w = int(length * W)
        cut_h = int(length * H)
        cutmix_lambda = (cut_w * cut_h) / (H * W)

    # Create the new inputs.
    X_cutmix = torch.clone(input)
    # Sample a rectangular box using lambda. Use variable names from the paper.
    if bbox:
        rx, ry, rw, rh = bbox[0], bbox[1], bbox[2], bbox[3]
        box_area = (rw - rx) * (rh - ry)
        cutmix_lambda = box_area / (H * W)
    else:
        rx, ry, rw, rh = _rand_bbox(input.shape[2], input.shape[3], cutmix_lambda, uniform_sampling=uniform_sampling)
        bbox = (rx, ry, rw, rh)

    # Fill in the box with a part of a random image.
    X_cutmix[:, :, rx:rw, ry:rh] = X_cutmix[shuffled_idx, :, rx:rw, ry:rh]
    # adjust lambda to exactly match pixel ratio. This is an implementation detail taken from
    # the original implementation, and implies lambda is not actually beta distributed.
    adjusted_lambda = _adjust_lambda(cutmix_lambda, input, bbox)

    # Make a shuffled version of y for interpolation
    y_shuffled = target[shuffled_idx]
    # Interpolate between labels using the adjusted lambda
    # First check if labels are indices. If so, convert them to onehots.
    # This is under the assumption that the loss expects torch.LongTensor, which is true for pytorch cross_entropy
    if _check_for_index_targets(target):
        y_onehot = F.one_hot(target, num_classes=num_classes)
        y_shuffled_onehot = F.one_hot(y_shuffled, num_classes=num_classes)
        y_cutmix = adjusted_lambda * y_onehot + (1 - adjusted_lambda) * y_shuffled_onehot
    else:
        y_cutmix = adjusted_lambda * target + (1 - adjusted_lambda) * y_shuffled

    return X_cutmix, y_cutmix


class CutMix(Algorithm):
    """`CutMix <https://arxiv.org/abs/1905.04899>`_ trains the network on non-overlapping combinations of pairs of
    examples and interpolated targets rather than individual examples and targets.

    This is done by taking a non-overlapping combination of a given batch X with a
    randomly permuted copy of X. The area is drawn from a ``Beta(alpha, alpha)``
    distribution.

    Training in this fashion sometimes reduces generalization error.

    Args:
        num_classes (int): the number of classes in the task labels.
        alpha (float, optional): the psuedocount for the Beta distribution
            used to sample area parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair. Default: ``1``.
        uniform_sampling (bool, optional): If ``True``, sample the bounding
            box such that each pixel has an equal probability of being mixed.
            If ``False``, defaults to the sampling used in the original
            paper implementation. Default: ``False``.

    Example:
        .. testcode::

            from composer.algorithms import CutMix
            algorithm = CutMix(num_classes=10, alpha=0.2)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(self, num_classes: int, alpha: float = 1., uniform_sampling: bool = False):
        self.num_classes = num_classes
        self.alpha = alpha
        self._uniform_sampling = uniform_sampling
        self._indices = torch.Tensor()
        self._cutmix_lambda = 0.0
        self._bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

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
        _cutmix_lambda = _gen_cutmix_coef(alpha)
        self._bbox = _rand_bbox(input.shape[2], input.shape[3], _cutmix_lambda, uniform_sampling=self._uniform_sampling)
        self._cutmix_lambda = _adjust_lambda(_cutmix_lambda, input, self._bbox)

        new_input, new_target = cutmix_batch(
            input=input,
            target=target,
            num_classes=self.num_classes,
            alpha=alpha,
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
        alpha: Parameter for the ``Beta(alpha, alpha)`` distribution

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
               cy: Optional[int] = None,
               uniform_sampling: bool = False) -> Tuple[int, int, int, int]:
    """Randomly samples a bounding box with area determined by cutmix_lambda.

    Adapted from original implementation https://github.com/clovaai/CutMix-PyTorch

    Args:
        W: Width of the image
        H: Height of the image
        cutmix_lambda: Lambda param from cutmix, used to set the area of the
            box if ``cut_w`` or ``cut_h`` is not provided.
        cx: Optional x coordinate of the center of the box.
        cy: Optional y coordinate of the center of the box.
        cut_w: Optional width of the box
        cut_h: Optional height of the box
        uniform_sampling: If true, sample the bounding box such that each pixel
            has an equal probability of being mixed. If false, defaults to the
            sampling used in the original paper implementation.

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
        if uniform_sampling is True:
            cx = np.random.randint(-cut_w // 2, high=W + cut_w // 2)
        else:
            cx = np.random.randint(W)
    if cy is None:
        if uniform_sampling is True:
            cy = np.random.randint(-cut_h // 2, high=H + cut_h // 2)
        else:
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
