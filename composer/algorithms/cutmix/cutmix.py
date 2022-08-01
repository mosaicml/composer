# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CutMix classes and functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.loss.utils import ensure_targets_one_hot

log = logging.getLogger(__name__)

__all__ = ['CutMix', 'cutmix_batch']


def cutmix_batch(input: Tensor,
                 target: Tensor,
                 length: Optional[float] = None,
                 alpha: float = 1.,
                 bbox: Optional[Tuple] = None,
                 indices: Optional[torch.Tensor] = None,
                 uniform_sampling: bool = False) -> Tuple[torch.Tensor, torch.Tensor, float, Tuple]:
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
    The original paper uses a fixed value of ``alpha = 1``.

    Alternatively, one may provide a bounding box to mask directly, in
    which case ``alpha`` is ignored and ``length`` must not be provided.

    The same masked region is used for the whole batch.

    .. note::
        The masked region is clipped at the spatial boundaries of the inputs.
        This means that there is no padding required, but the actual region
        used may be smaller than the nominal size computed using ``length``
        or ``alpha``.

    Args:
        input (torch.Tensor): input tensor of shape ``(N, C, H, W)``.
        target (torch.Tensor): target tensor of either shape ``N`` or
            ``(N, num_classes)``. In the former case, elements of ``target``
            must be integer class ids in the range ``0..num_classes``. In the
            latter case, rows of ``target`` may be arbitrary vectors of targets,
            including, e.g., one-hot encoded class labels, smoothed class
            labels, or multi-output regression targets.
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
        target_perm (torch.Tensor): The labels of the mixed-in examples
        area (float): The fractional area of the unmixed region.
        bounding_box (tuple): the ``(left, top, right, bottom)`` coordinates of
            the bounding box that defines the mixed region.

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
            X_mixed, target_perm, area, _ = cutmix_batch(X, y, alpha=0.2)
    """
    if bbox is not None and length is not None:
        raise ValueError(f'Cannot provide both length and bbox; got {length} and {bbox}')

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
    adjusted_lambda = _adjust_lambda(input, bbox)

    # Make a shuffled version of y for interpolation
    y_shuffled = target[shuffled_idx]

    return X_cutmix, y_shuffled, adjusted_lambda, bbox


class CutMix(Algorithm):
    """`CutMix <https://arxiv.org/abs/1905.04899>`_ trains the network on non-overlapping combinations
    of pairs of examples and interpolated targets rather than individual examples and targets.

    This is done by taking a non-overlapping combination of a given batch X with a
    randomly permuted copy of X. The area is drawn from a ``Beta(alpha, alpha)``
    distribution.

    Training in this fashion sometimes reduces generalization error.

    Args:
        alpha (float, optional): the psuedocount for the Beta distribution
            used to sample area parameters. As ``alpha`` grows, the two samples
            in each pair tend to be weighted more equally. As ``alpha``
            approaches 0 from above, the combination approaches only using
            one element of the pair. Default: ``1``.
        interpolate_loss (bool, optional): Interpolates the loss rather than the labels.
            A useful trick when using a cross entropy loss. Will produce incorrect behavior
            if the loss is not a linear function of the targets. Default: ``False``
        uniform_sampling (bool, optional): If ``True``, sample the bounding
            box such that each pixel has an equal probability of being mixed.
            If ``False``, defaults to the sampling used in the original
            paper implementation. Default: ``False``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.

    Example:
        .. testcode::

            from composer.algorithms import CutMix
            algorithm = CutMix(alpha=0.2)
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
        alpha: float = 1.,
        interpolate_loss: bool = False,
        uniform_sampling: bool = False,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        self.alpha = alpha
        self.interpolate_loss = interpolate_loss
        self._uniform_sampling = uniform_sampling

        self._indices = torch.Tensor()
        self._cutmix_lambda = 0.0
        self._bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._permuted_target = torch.Tensor()
        self._adjusted_lambda = 0.0
        self.input_key, self.target_key = input_key, target_key

    def match(self, event: Event, state: State) -> bool:
        if self.interpolate_loss:
            return event in [Event.BEFORE_FORWARD, Event.BEFORE_BACKWARD]
        else:
            return event in [Event.BEFORE_FORWARD, Event.BEFORE_LOSS]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        input = state.batch_get_item(key=self.input_key)
        target = state.batch_get_item(key=self.target_key)

        if not isinstance(input, torch.Tensor):
            raise NotImplementedError('Multiple tensors for inputs not supported yet.')
        if not isinstance(target, torch.Tensor):
            raise NotImplementedError('Multiple tensors for targets not supported yet.')

        alpha = self.alpha

        if event == Event.BEFORE_FORWARD:
            # these are saved only for testing
            self._indices = _gen_indices(input)

            _cutmix_lambda = _gen_cutmix_coef(alpha)
            self._bbox = _rand_bbox(input.shape[2],
                                    input.shape[3],
                                    _cutmix_lambda,
                                    uniform_sampling=self._uniform_sampling)
            self._adjusted_lambda = _adjust_lambda(input, self._bbox)

            new_input, self._permuted_target, _, _ = cutmix_batch(input=input,
                                                                  target=target,
                                                                  alpha=self.alpha,
                                                                  bbox=self._bbox,
                                                                  indices=self._indices,
                                                                  uniform_sampling=self._uniform_sampling)

            state.batch_set_item(key=self.input_key, value=new_input)

        if not self.interpolate_loss and event == Event.BEFORE_LOSS:
            # Interpolate the targets
            if not isinstance(state.outputs, torch.Tensor):
                raise NotImplementedError('Multiple output tensors not supported yet')
            if not isinstance(target, torch.Tensor):
                raise NotImplementedError('Multiple target tensors not supported yet')
            if self._permuted_target.ndim > 2 and self._permuted_target.shape[-2:] == input.shape[-2:]:
                # Target has the same height and width as the input, no need to interpolate.
                x1, y1, x2, y2 = self._bbox
                target[..., x1:x2, y1:y2] = self._permuted_target[..., x1:x2, y1:y2]
            else:
                # Need to interpolate on dense/one-hot targets.
                target = ensure_targets_one_hot(state.outputs, target)
                permuted_target = ensure_targets_one_hot(state.outputs, self._permuted_target)
                # Interpolate to get the new target
                target = self._adjusted_lambda * target + (1 - self._adjusted_lambda) * permuted_target
            # Create the new batch
            state.batch_set_item(key=self.target_key, value=target)

        if self.interpolate_loss and event == Event.BEFORE_BACKWARD:
            if self._permuted_target.ndim > 2 and self._permuted_target.shape[-2:] == input.shape[-2:]:
                raise ValueError("Can't interpolate loss when target has the same height and width as the input")

            # Grab the loss function
            if hasattr(state.model, 'loss'):
                loss_fn = state.model.loss
            elif hasattr(state.model, 'module') and hasattr(state.model.module, 'loss'):
                if isinstance(state.model.module, torch.nn.Module):
                    loss_fn = state.model.module.loss
                else:
                    raise TypeError('state.model.module must be a torch module')
            else:
                raise AttributeError('Loss must be accessible via model.loss or model.module.loss')
            # Verify that the loss is callable
            if not callable(loss_fn):
                raise TypeError('Loss must be callable')
            # Interpolate the loss
            new_loss = loss_fn(state.outputs, (input, self._permuted_target))
            if not isinstance(state.loss, torch.Tensor):
                raise NotImplementedError('Multiple losses not supported yet')
            if not isinstance(new_loss, torch.Tensor):
                raise NotImplementedError('Multiple losses not supported yet')
            state.loss = self._adjusted_lambda * state.loss + (1 - self._adjusted_lambda) * new_loss


def _gen_indices(x: Tensor) -> Tensor:
    """Generates indices of a random permutation of elements of a batch.

    Args:
        x (torch.Tensor): input tensor of shape ``(B, d1, d2, ..., dn)``,
            B is batch size, d1-dn are feature dimensions.

    Returns:
        indices: A random permutation of the batch indices.
    """
    return torch.randperm(x.shape[0])


def _gen_cutmix_coef(alpha: float) -> float:
    """Generates lambda from ``Beta(alpha, alpha)``.

    Args:
        alpha (float): Parameter for the ``Beta(alpha, alpha)`` distribution.

    Returns:
        cutmix_lambda: Lambda parameter for performing CutMix.
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
    """Randomly samples a bounding box with area determined by ``cutmix_lambda``.

    Adapted from original implementation https://github.com/clovaai/CutMix-PyTorch

    Args:
        W (int): Width of the image
        H (int): Height of the image
        cutmix_lambda (float): Lambda param from cutmix, used to set the area of the
            box if ``cut_w`` or ``cut_h`` is not provided.
        cx (int, optional): Optional x coordinate of the center of the box.
        cy (int, optional): Optional y coordinate of the center of the box.
        uniform_sampling (bool, optional): If true, sample the bounding box such that each pixel
            has an equal probability of being mixed. If false, defaults to the
            sampling used in the original paper implementation. Default: ``False``.

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


def _adjust_lambda(x: Tensor, bbox: Tuple) -> float:
    """Rescale the cutmix lambda according to the size of the clipped bounding box.

    Args:
        x (torch.Tensor): input tensor of shape ``(B, d1, d2, ..., dn)``, B is batch size, d1-dn
            are feature dimensions.
        bbox (tuple): (x1, y1, x2, y2) coordinates of the boundind box, obeying x2 > x1, y2 > y1.

    Returns:
        adjusted_lambda: Rescaled cutmix_lambda to account for part of the bounding box
            being potentially out of bounds of the input.
    """
    rx, ry, rw, rh = bbox[0], bbox[1], bbox[2], bbox[3]
    adjusted_lambda = 1 - ((rw - rx) * (rh - ry) / (x.size()[-1] * x.size()[-2]))
    return adjusted_lambda
