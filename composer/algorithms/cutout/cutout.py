# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CutOut classes and functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np
import torch
from PIL.Image import Image as PillowImage
from torch import Tensor

from composer.algorithms.utils.augmentation_common import image_as_type
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['CutOut', 'cutout_batch']

ImgT = TypeVar('ImgT', torch.Tensor, PillowImage)


def cutout_batch(input: ImgT, num_holes: int = 1, length: float = 0.5, uniform_sampling: bool = False) -> ImgT:
    """See :class:`.CutOut`.

    Args:
        input (PIL.Image.Image | torch.Tensor): Image or batch of images. If
            a :class:`torch.Tensor`, must be a single image of shape ``(C, H, W)``
            or a batch of images of shape ``(N, C, H, W)``.
        num_holes: Integer number of holes to cut out. Default: ``1``.
        length (float, optional): Relative side length of the masked region.
            If specified, ``length`` is interpreted as a fraction of ``H`` and
            ``W``, and the resulting box is a square with side length
            ``length * min(H, W)``. Must be in the interval :math:`(0, 1)`.
            Default: ``0.5``.
        uniform_sampling (bool, optional): If ``True``, sample the bounding
            box such that each pixel has an equal probability of being masked.
            If ``False``, defaults to the sampling used in the original paper
            implementation. Default: ``False``.

    Returns:
        X_cutout: Batch of images with ``num_holes`` square holes with
            dimension determined by ``length`` replaced with zeros.

    Example:
         .. testcode::

            from composer.algorithms.cutout import cutout_batch
            new_input_batch = cutout_batch(X_example, num_holes=1, length=0.25)
    """
    X_tensor = image_as_type(input, torch.Tensor)
    h = X_tensor.shape[-2]
    w = X_tensor.shape[-1]

    length = int(min(h, w) * length)

    mask = torch.ones_like(X_tensor)
    for _ in range(num_holes):
        if uniform_sampling is True:
            y = np.random.randint(-length // 2, high=h + length // 2)
            x = np.random.randint(-length // 2, high=w + length // 2)
        else:
            y = np.random.randint(h)
            x = np.random.randint(w)

        mask = _generate_mask(mask, w, h, x, y, length)

    X_cutout = X_tensor * mask
    X_out = image_as_type(X_cutout, input.__class__)  # pyright struggling with unions
    return X_out


class CutOut(Algorithm):
    """`CutOut <https://arxiv.org/abs/1708.04552>`_ is a data augmentation technique
    that works by masking out one or more square regions of an input image.

    This implementation cuts out the same square from all images in a batch.

    Example:
         .. testcode::

            from composer.algorithms import CutOut
            from composer.trainer import Trainer

            cutout_algorithm = CutOut(num_holes=1, length=0.25)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[cutout_algorithm],
                optimizers=[optimizer]
            )

    Args:
        num_holes (int, optional): Integer number of holes to cut out.
            Default: ``1``.
        length (float, optional): Relative side length of the masked region.
            If specified, ``length`` is interpreted as a fraction of ``H`` and
            ``W``, and the resulting box is a square with side length
            ``length * min(H, W)``. Must be in the interval :math:`(0, 1)`.
            Default: ``0.5``.
        uniform_sampling (bool, optional): If ``True``, sample the bounding
            box such that each pixel has an equal probability of being masked.
            If ``False``, defaults to the sampling used in the original paper
            implementation. Default: ``False``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
    """

    def __init__(self,
                 num_holes: int = 1,
                 length: float = 0.5,
                 uniform_sampling: bool = False,
                 input_key: Union[str, int, Callable, Any] = 0):
        self.num_holes = num_holes
        self.length = length
        self.uniform_sampling = uniform_sampling
        self.input_key = input_key

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        x = state.batch_get_item(self.input_key)
        assert isinstance(x, Tensor), 'Multiple tensors not supported for Cutout.'

        new_x = cutout_batch(x, num_holes=self.num_holes, length=self.length, uniform_sampling=self.uniform_sampling)
        state.batch_set_item(self.input_key, new_x)


def _generate_mask(mask: Tensor, width: int, height: int, x: int, y: int, cutout_length: int) -> Tensor:
    y1 = np.clip(y - cutout_length // 2, 0, height)
    y2 = np.clip(y + cutout_length // 2, 0, height)
    x1 = np.clip(x - cutout_length // 2, 0, width)
    x2 = np.clip(x + cutout_length // 2, 0, width)

    mask[..., y1:y2, x1:x2] = 0.

    return mask
