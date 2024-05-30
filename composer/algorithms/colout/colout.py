# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core ColOut classes and functions."""

from __future__ import annotations

import logging
import textwrap
import weakref
from typing import Any, Callable, TypeVar, Union

import torch
import torch.utils.data
from PIL.Image import Image as PillowImage
from torch import Tensor
from torchvision.datasets import VisionDataset

from composer.algorithms.utils.augmentation_common import image_as_type
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import add_vision_dataset_transform, ensure_tuple

log = logging.getLogger(__name__)

ImgT = TypeVar('ImgT', torch.Tensor, PillowImage)

__all__ = ['ColOut', 'ColOutTransform', 'colout_batch']


def colout_batch(
    sample: Union[ImgT, tuple[ImgT, ImgT]],
    p_row: float = 0.15,
    p_col: float = 0.15,
    resize_target: Union[bool, str] = 'auto',
) -> Union[torch.Tensor, ImgT, tuple[Tensor, Tensor], tuple[ImgT, ImgT]]:
    """Applies ColOut augmentation to a batch of images and (optionally) targets,
    dropping the same random rows and columns from all images and targets in a batch.

    See the :doc:`Method Card </method_cards/colout>` for more details.

    Example:
         .. testcode::

            from composer.algorithms.colout import colout_batch
            new_X = colout_batch(X_example, p_row=0.15, p_col=0.15)

    Args:
        sample (torch.Tensor | PIL.Image | tuple[torch.Tensor, torch.Tensor] | tuple[PIL.Image, PIL.Image]):
            Either a single tensor or image or a 2-tuple of tensors or images. When tensor(s), the tensor must be of shape
            ``CHW`` for a single image or ``NCHW`` for a batch of images of shape.
        p_row (float, optional): Fraction of rows to drop (drop along H). Default: ``0.15``.
        p_col (float, optional): Fraction of columns to drop (drop along W). Default: ``0.15``.
        resize_target (bool | str, optional): If ``sample`` is a tuple, whether to resize both objects in the tuple.
            If set to ``'auto'``, both objects will be resized if they have the same spatial dimensions.
            Otherwise, only the first object is resized. Default: ``'auto'``.

    Returns:
        torch.Tensor | PIL.Image | tuple[torch.Tensor, torch.Tensor] | tuple[PIL.Image, PIL.Image]:
                A smaller image or 2-tuple of images with random rows and columns dropped.
    """

    sample = ensure_tuple(sample)
    if len(sample) > 2:
        raise ValueError('sample must either be single object or a tuple with a max length of 2')
    input = sample[0]

    # Convert image to Tensor if needed
    X_tensor = image_as_type(input, torch.Tensor)

    # Get the dimensions of the image
    row_size = X_tensor.shape[-2]
    col_size = X_tensor.shape[-1]

    # Determine how many rows and columns to keep
    kept_row_size = int((1 - p_row) * row_size)
    kept_col_size = int((1 - p_col) * col_size)

    # Randomly choose indices to keep. Must be sorted for slicing
    kept_row_idx = sorted(torch.randperm(row_size)[:kept_row_size].numpy())
    kept_col_idx = sorted(torch.randperm(col_size)[:kept_col_size].numpy())

    # Keep only the selected row and columns
    X_colout = X_tensor[..., kept_row_idx, :]
    X_colout = X_colout[..., :, kept_col_idx]

    # convert back to same type as input, and strip added batch dim if needed;
    # we can't just reshape to input shape because we've reduced the spatial size
    if not isinstance(input, torch.Tensor) or (input.ndim < X_colout.ndim):
        X_colout = X_colout.reshape(X_colout.shape[-3:])
    X_colout = image_as_type(X_colout, type(input))

    resize_target = _should_resize_target(sample, resize_target)

    if resize_target:
        target = sample[1]
        Y_tensor = image_as_type(target, torch.Tensor)
        Y_colout = Y_tensor[..., kept_row_idx, :]
        Y_colout = Y_colout[..., :, kept_col_idx]

        # convert back to same type as input, and strip added batch dim if needed;
        # we can't just reshape to input shape because we've reduced the spatial size
        if not isinstance(target, torch.Tensor) or (target.ndim < Y_colout.ndim):
            Y_colout = Y_colout.reshape(Y_colout.shape[-3:])
        Y_colout = image_as_type(Y_colout, type(target))

        return X_colout, Y_colout

    return X_colout


class ColOutTransform:
    """Torchvision-like transform for performing the ColOut augmentation,
    where random rows and columns are dropped from up to two Torch tensors
    or two PIL images.

    See the :doc:`Method Card </method_cards/colout>` for more details.

    Example:
         .. testcode::

            from torchvision import datasets, transforms
            from composer.algorithms.colout import ColOutTransform
            colout_transform = ColOutTransform(p_row=0.15, p_col=0.15)
            transforms = transforms.Compose([colout_transform, transforms.ToTensor()])

    Args:
        p_row (float, optional): Fraction of rows to drop (drop along H). Default: ``0.15``.
        p_col (float, optional): Fraction of columns to drop (drop along W). Default: ``0.15``.
        resize_target (bool | str, optional): Whether to resize the target in addition to the input.
            If set to ``'auto'``, resizing the target will be based on if the target has the same spatial
            dimensions as the input. Default: ``'auto'``.
    """

    def __init__(self, p_row: float = 0.15, p_col: float = 0.15, resize_target: Union[bool, str] = 'auto'):
        self.p_row = p_row
        self.p_col = p_col
        self.resize_target = resize_target

    def __call__(
        self,
        sample: Union[ImgT, tuple[ImgT, ImgT]],
    ) -> Union[torch.Tensor, ImgT, tuple[Tensor, Tensor], tuple[ImgT, ImgT]]:
        """Drops random rows and columns from up to two images.

        Args:
            sample (torch.Tensor | PIL.Image | tuple[torch.Tensor, torch.Tensor] | tuple[PIL.Image, PIL.Image]):
                A single image or a 2-tuple of images as either :class:`torch.Tensor` or :class:`PIL.Image`.

        Returns:
            torch.Tensor | PIL.Image | tuple[torch.Tensor, torch.Tensor] | tuple[PIL.Image, PIL.Image]:
                A smaller image or 2-tuple of images with random rows and columns dropped.
        """

        sample = ensure_tuple(sample)
        if len(sample) > 2:
            raise ValueError(f'Colout transform does not support sample tuple of length {len(sample)} > 2')

        return colout_batch(sample, p_row=self.p_row, p_col=self.p_col, resize_target=self.resize_target)


class ColOut(Algorithm):
    """Drops a fraction of the rows and columns of an input image and (optionally) a target image. If the fraction of
    rows/columns dropped isn't too large, this does not significantly alter the content of the image, but reduces its
    size and provides extra variability.

    If ``batch`` is True (the default), then this algorithm runs on :attr:`.Event.AFTER_DATALOADER`
    to modify the batch.

    Otherwise, if ``batch=False`` (the default), this algorithm runs on :attr:`.Event.INIT` to insert
    a dataset transformation. It is a no-op if this algorithm already applied itself on the :attr:`State.train_dataloader.dataset`.

    See the :doc:`Method Card </method_cards/colout>` for more details.

    Example:
         .. testcode::

            from composer.algorithms import ColOut
            from composer.trainer import Trainer
            colout_algorithm = ColOut(p_row=0.15, p_col=0.15, batch=True)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[colout_algorithm],
                optimizers=[optimizer]
            )

    Args:
        p_row (float, optional): Fraction of rows to drop (drop along H). Default: ``0.15``.
        p_col (float, optional): Fraction of columns to drop (drop along W). Default: ``0.15``.
        batch (bool, optional): Run ColOut at the batch level. Default: ``True``.
        resize_target (bool | str, optional): Whether to resize the target in addition to the input. If set to ``'auto'``, resizing
            the target will be based on if the target has the same spatial dimensions as the input. Default: ``auto``.
        input_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
        target_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.
    """

    def __init__(
        self,
        p_row: float = 0.15,
        p_col: float = 0.15,
        batch: bool = True,
        resize_target: Union[bool, str] = 'auto',
        input_key: Union[str, int, tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, tuple[Callable, Callable], Any] = 1,
    ):
        if not (0 <= p_col <= 1):
            raise ValueError('p_col must be between 0 and 1')

        if not (0 <= p_row <= 1):
            raise ValueError('p_row must be between 0 and 1')

        if (not isinstance(resize_target, bool)) and (isinstance(resize_target, str) and resize_target != 'auto'):
            raise ValueError(f'resize_target must be a boolean or ``auto``. Received: {resize_target}')

        if resize_target is True and batch is False:
            raise NotImplementedError(f'Resizing targets is not currently support with batch=``False``')

        self.p_row = p_row
        self.p_col = p_col
        self.batch = batch
        self.resize_target = resize_target
        self._transformed_datasets = weakref.WeakSet()
        self.input_key, self.target_key = input_key, target_key

    def match(self, event: Event, state: State) -> bool:
        if self.batch:
            return event == Event.AFTER_DATALOADER
        else:
            if event != Event.FIT_START:
                return False
            assert state.dataloader is not None, 'dataloader should be defined on fit start'
            if not isinstance(state.dataloader, torch.utils.data.DataLoader):
                raise TypeError(f'{type(self).__name__} requires a PyTorch dataloader.')
            return state.dataloader.dataset not in self._transformed_datasets

    def _apply_sample(self, state: State) -> None:
        """Add the ColOut dataset transform to the dataloader."""
        assert isinstance(state.dataloader, torch.utils.data.DataLoader), 'dataloader type checked on match()'
        dataset = state.dataloader.dataset

        transform = ColOutTransform(p_row=self.p_row, p_col=self.p_col, resize_target=self.resize_target)

        if not isinstance(dataset, VisionDataset):
            raise TypeError(
                textwrap.dedent(
                    f"""\
                To use {type(self).__name__}, the dataset must be a
                {VisionDataset.__qualname__}, not {type(dataset).__name__}""",
                ),
            )
        add_vision_dataset_transform(dataset, transform, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)

    def _apply_batch(self, state: State) -> None:
        """Transform a batch of images using the ColOut augmentation."""
        inputs, target = state.batch_get_item(key=self.input_key), state.batch_get_item(key=self.target_key)
        assert isinstance(inputs, Tensor) and isinstance(target, Tensor), \
            'Inputs and target must be of type torch.Tensor for batch-wise ColOut'

        sample = (inputs, target)
        resize_target = _should_resize_target(sample, resize_target=self.resize_target)
        colout_result = colout_batch(sample, p_row=self.p_row, p_col=self.p_col, resize_target=resize_target)

        # colout_result will be a tuple if the targets are resized and a single object otherwise
        if resize_target:
            new_input, new_target = colout_result
            state.batch_set_item(self.input_key, new_input)
            state.batch_set_item(self.target_key, new_target)
        else:
            new_input = colout_result
            state.batch_set_item(self.input_key, new_input)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if self.batch:
            self._apply_batch(state)
        else:
            self._apply_sample(state)


def _should_resize_target(sample: Union[ImgT, tuple[ImgT, ImgT]], resize_target: Union[bool, str]) -> bool:
    """Helper function to determine if both objects in the tuple should be resized.

    Decision is based on ``resize_target`` and if both objects in the tuple have the same spatial size.
    """

    sample = ensure_tuple(sample)
    if len(sample) > 2:
        raise ValueError('sample must either be single object or a tuple with a max length of 2')
    input = sample[0]

    if isinstance(resize_target, bool):
        return resize_target

    if len(sample) == 1:
        return False

    if isinstance(resize_target, str) and resize_target.lower() == 'auto':
        input_size = input.shape[-2:] if isinstance(input, torch.Tensor) else input.size[::-1]
        target = sample[1]
        if isinstance(target, PillowImage):
            return target.size[::-1] == input_size
        else:
            return target.ndim > 2 and target.shape[-2:] == input_size

    raise ValueError("resize_target must either be a boolean or 'auto'")
