# Copyright 2021 MosaicML. All Rights Reserved.

"""Core ColOut classes and functions."""

from __future__ import annotations

import logging
import textwrap
import weakref
from typing import TypeVar

import torch
from PIL.Image import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as TF

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Tensor
from composer.utils.data import add_dataset_transform

log = logging.getLogger(__name__)

TImg = TypeVar("TImg", torch.Tensor, Image)

__all__ = ["ColOut", "ColOutTransform", "colout_image", "colout_batch"]


def colout_image(img: TImg, p_row: float = 0.15, p_col: float = 0.15) -> TImg:
    """Drops random rows and columns from a single image.

    Example:
         .. testcode::

            from composer.algorithms.colout import colout_image
            new_image = colout_image(
                img=image,
                p_row=0.15,
                p_col=0.15
            )

    Args:
        img (torch.Tensor or PIL Image): An input image as a torch.Tensor or PIL image
        p_row (float): Fraction of rows to drop (drop along H).
        p_col (float): Fraction of columns to drop (drop along W).

    Returns:
        torch.Tensor or PIL Image: A smaller image with rows and columns dropped
    """

    # Convert image to Tensor if needed
    if isinstance(img, Image):
        img_tensor = TF.to_tensor(img)
    else:
        img_tensor = img

    # Get the dimensions of the image
    row_size = img_tensor.shape[1]
    col_size = img_tensor.shape[2]

    # Determine how many rows and columns to keep
    kept_row_size = int((1 - p_row) * row_size)
    kept_col_size = int((1 - p_col) * col_size)

    # Randomly choose indices to keep. Must be sorted for slicing
    kept_row_idx = sorted(torch.randperm(row_size)[:kept_row_size].numpy())
    kept_col_idx = sorted(torch.randperm(col_size)[:kept_col_size].numpy())

    # Keep only the selected row and columns
    img_tensor = img_tensor[:, kept_row_idx, :]
    img_tensor = img_tensor[:, :, kept_col_idx]

    # Convert back to PIL for the rest of the augmentation pipeline
    if isinstance(img, Image):
        return TF.to_pil_image(img_tensor)
    else:
        return img_tensor


class ColOutTransform:
    """Torchvision-like transform for performing the ColOut augmentation, where random rows and columns are dropped from
    a single image.

    Example:
         .. testcode::

            from torchvision import datasets, transforms
            from composer.algorithms.colout import ColOutTransform
            colout_transform = ColOutTransform(p_row=0.15, p_col=0.15)
            transforms = transforms.Compose([colout_transform, transforms.ToTensor()])

    Args:
        p_row (float): Fraction of rows to drop (drop along H).
        p_col (float): Fraction of columns to drop (drop along W).
    """

    def __init__(self, p_row: float = 0.15, p_col: float = 0.15):
        self.p_row = p_row
        self.p_col = p_col

    def __call__(self, img: TImg) -> TImg:
        """Drops random rows and columns from a single image.

        Args:
            img (torch.Tensor or PIL Image): An input image as a torch.Tensor or PIL image

        Returns:
            torch.Tensor or PIL Image: A smaller image with rows and columns dropped
        """
        return colout_image(img, self.p_row, self.p_col)


def colout_batch(X: torch.Tensor, p_row: float = 0.15, p_col: float = 0.15) -> torch.Tensor:
    """Applies ColOut augmentation to a batch of images, dropping the same random rows and columns from all images in a
    batch.

    Example:
         .. testcode::

            from composer.algorithms.colout import colout_batch
            new_input_batch = colout_batch(
                X=input_batch,
                p_row=0.15,
                p_col=0.15
            )

    Args:
        X: Batch of images of shape (N, C, H, W).
        p_row: Fraction of rows to drop (drop along H).
        p_col: Fraction of columns to drop (drop along W).

    Returns:
        torch.Tensor: Input batch tensor with randomly dropped columns and rows.
    """

    # Get the dimensions of the image
    row_size = X.shape[2]
    col_size = X.shape[3]

    # Determine how many rows and columns to keep
    kept_row_size = int((1 - p_row) * row_size)
    kept_col_size = int((1 - p_col) * col_size)

    # Randomly choose indices to keep. Must be sorted for slicing
    kept_row_idx = sorted(torch.randperm(row_size)[:kept_row_size].numpy())
    kept_col_idx = sorted(torch.randperm(col_size)[:kept_col_size].numpy())

    # Keep only the selected row and columns
    X_colout = X[:, :, kept_row_idx, :]
    X_colout = X_colout[:, :, :, kept_col_idx]
    return X_colout


class ColOut(Algorithm):
    """Drops a fraction of the rows and columns of an input image. If the fraction of rows/columns dropped isn't too
    large, this does not significantly alter the content of the image, but reduces its size and provides extra
    variability.

    If ``batch`` is True (the default), this algorithm runs on :attr:`Event.INIT` to insert a dataset transformation.
    It is a no-op if this algorithm already applied itself on the :attr:`State.train_dataloader.dataset`.

    Otherwise, if ``batch`` is False, then this algorithm runs on :attr:`Event.AFTER_DATALOADER` to modify the batch.

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
        p_row (float): Fraction of rows to drop (drop along H).
        p_col (float): Fraction of columns to drop (drop along W).
        batch (bool): Run ColOut at the batch level.
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
        add_dataset_transform(dataset, transform, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)

    def _apply_batch(self, state: State) -> None:
        """Transform a batch of images using the ColOut augmentation."""
        inputs, labels = state.batch_pair
        assert isinstance(inputs, Tensor), "Multiple Tensors not supported yet for ColOut"
        new_inputs = colout_batch(inputs, p_row=self.p_row, p_col=self.p_col)

        state.batch = (new_inputs, labels)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Applies ColOut augmentation to the state's input.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Optional[Logger], optional): the training logger. Defaults to None.
        """
        if self.batch:
            self._apply_batch(state)
        else:
            self._apply_sample(state)
