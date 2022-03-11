# Copyright 2021 MosaicML. All Rights Reserved.

"""Core ColOut classes and functions."""

from __future__ import annotations

import logging
import textwrap
import weakref
from typing import TypeVar

import torch
from PIL.Image import Image as PillowImage
from torchvision.datasets import VisionDataset

from composer.algorithms.utils.augmentation_common import image_as_type
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Tensor
from composer.datasets.utils import add_vision_dataset_transform

log = logging.getLogger(__name__)

ImgT = TypeVar("ImgT", torch.Tensor, PillowImage)

__all__ = ["ColOut", "ColOutTransform", "colout_batch"]


def colout_batch(input: ImgT, p_row: float = 0.15, p_col: float = 0.15) -> ImgT:
    """Applies ColOut augmentation to a batch of images, dropping the same random rows and columns from all images in a
    batch.

    See the :doc:`Method Card </method_cards/colout>` for more details.

    Example:
         .. testcode::

            from composer.algorithms.colout import colout_batch
            new_X = colout_batch(X_example, p_row=0.15, p_col=0.15)

    Args:
        input: :class:`PIL.Image.Image` or :class:`torch.Tensor` of image data. In
            the latter case, must be a single image of shape ``CHW`` or a batch
            of images of shape ``NCHW``.
        p_row: Fraction of rows to drop (drop along H). Default: ``0.15``.
        p_col: Fraction of columns to drop (drop along W). Default: ``0.15``.

    Returns:
        torch.Tensor: Input batch tensor with randomly dropped columns and rows.
    """

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
    return X_colout


class ColOutTransform:
    """Torchvision-like transform for performing the ColOut augmentation, where random rows and columns are dropped from
    a single image.

    See the :doc:`Method Card </method_cards/colout>` for more details.

    Example:
         .. testcode::

            from torchvision import datasets, transforms
            from composer.algorithms.colout import ColOutTransform
            colout_transform = ColOutTransform(p_row=0.15, p_col=0.15)
            transforms = transforms.Compose([colout_transform, transforms.ToTensor()])

    Args:
        p_row (float): Fraction of rows to drop (drop along H). Default: ``0.15``.
        p_col (float): Fraction of columns to drop (drop along W). Default: ``0.15``.
    """

    def __init__(self, p_row: float = 0.15, p_col: float = 0.15):
        self.p_row = p_row
        self.p_col = p_col

    def __call__(self, img: ImgT) -> ImgT:
        """Drops random rows and columns from a single image.

        Args:
            img (torch.Tensor or PIL Image): An input image as a torch.Tensor or PIL image

        Returns:
            torch.Tensor or PIL Image: A smaller image with rows and columns dropped
        """
        return colout_batch(img, self.p_row, self.p_col)


class ColOut(Algorithm):
    """Drops a fraction of the rows and columns of an input image. If the fraction of rows/columns dropped isn't too
    large, this does not significantly alter the content of the image, but reduces its size and provides extra
    variability.

    If ``batch`` is True (the default), this algorithm runs on :attr:`Event.INIT` to insert a dataset transformation.
    It is a no-op if this algorithm already applied itself on the :attr:`State.train_dataloader.dataset`.

    Otherwise, if ``batch`` is False, then this algorithm runs on :attr:`Event.AFTER_DATALOADER` to modify the batch.

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
        p_row (float): Fraction of rows to drop (drop along H). Default: ``0.15``.
        p_col (float): Fraction of columns to drop (drop along W). Default: ``0.15``.
        batch (bool): Run ColOut at the batch level. Default: ``True``.
    """

    def __init__(self, p_row: float = 0.15, p_col: float = 0.15, batch: bool = True):
        if not (0 <= p_col <= 1):
            raise ValueError("p_col must be between 0 and 1")

        if not (0 <= p_row <= 1):
            raise ValueError("p_row must be between 0 and 1")

        self.p_row = p_row
        self.p_col = p_col
        self.batch = batch
        self._transformed_datasets = weakref.WeakSet()

    def match(self, event: Event, state: State) -> bool:
        if self.batch:
            return event == Event.AFTER_DATALOADER
        else:
            return event == Event.FIT_START and state.train_dataloader.dataset not in self._transformed_datasets

    def _apply_sample(self, state: State) -> None:
        """Add the ColOut dataset transform to the dataloader."""
        dataset = state.train_dataloader.dataset

        transform = ColOutTransform(p_row=self.p_row, p_col=self.p_col)

        if not isinstance(dataset, VisionDataset):
            raise TypeError(
                textwrap.dedent(f"""\
                To use {type(self).__name__}, the dataset must be a
                {VisionDataset.__qualname__}, not {type(dataset).__name__}"""))
        add_vision_dataset_transform(dataset, transform, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)

    def _apply_batch(self, state: State) -> None:
        """Transform a batch of images using the ColOut augmentation."""
        inputs, labels = state.batch_pair
        assert isinstance(inputs, Tensor), "Multiple Tensors not supported yet for ColOut"
        new_inputs = colout_batch(inputs, p_row=self.p_row, p_col=self.p_col)

        state.batch = (new_inputs, labels)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if self.batch:
            self._apply_batch(state)
        else:
            self._apply_sample(state)
