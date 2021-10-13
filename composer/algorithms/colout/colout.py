# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Union

import torch
import yahp as hp
from PIL.Image import Image
from torchvision.transforms import functional as TF

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Tensor
from composer.utils.data import add_dataset_transform

log = logging.getLogger(__name__)


def colout(img: Union[torch.Tensor, Image], p_row: float, p_col: float) -> Union[torch.Tensor, Image]:
    """Drops random rows and columns from a single image.

    Args:
        img (torch.Tensor or PIL Image): An input image as a torch.Tensor or PIL image
        p_row (float): Fraction of rows to drop (drop along H).
        p_col (float): Fraction of columns to drop (drop along W).

    Returns:
        torch.Tensor or PIL Image: A smaller image with rows and columns dropped
    """

    as_PIL = False

    # Convert image to Tensor if needed
    if isinstance(img, Image):
        as_PIL = True
        img = TF.to_tensor(img)

    if not isinstance(img, torch.Tensor):
        raise ValueError("Invalid input type: img must be either torch.Tensor or PIL Image.")

    # Get the dimensions of the image
    row_size = img.shape[1]
    col_size = img.shape[2]

    # Determine how many rows and columns to keep
    kept_row_size = int((1 - p_row) * row_size)
    kept_col_size = int((1 - p_col) * col_size)

    # Randomly choose indices to keep. Must be sorted for slicing
    kept_row_idx = sorted(torch.randperm(row_size)[:kept_row_size].numpy())
    kept_col_idx = sorted(torch.randperm(col_size)[:kept_col_size].numpy())

    # Keep only the selected row and columns
    img = img[:, kept_row_idx, :]
    img = img[:, :, kept_col_idx]

    # Convert back to PIL for the rest of the augmentation pipeline
    if as_PIL:
        return TF.to_pil_image(img)
    else:
        return img


class ColOutTransform:
    """ Torchvision-like transform for performing the ColOut augmentation, where random rows and columns are
        dropped from a single image.

    Args:
        p_row (float): Fraction of rows to drop (drop along H).
        p_col (float): Fraction of columns to drop (drop along W).
    """

    def __init__(self, p_row: float, p_col: float):
        self.p_row = p_row
        self.p_col = p_col

    def __call__(self, img: Union[torch.Tensor, Image]) -> Union[torch.Tensor, Image]:
        """ Drops random rows and columns from a single image.

        Args:
            img (torch.Tensor or PIL Image): An input image as a torch.Tensor or PIL image

        Returns:
            torch.Tensor or PIL Image: A smaller image with rows and columns dropped
        """
        return colout(img, self.p_row, self.p_col)


def batch_colout(X: torch.Tensor, p_row: float, p_col: float) -> torch.Tensor:
    """Applies ColOut augmentation to a batch of images, dropping the same
    random rows and columns from all images in a batch.

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


@dataclass
class ColOutHparams(AlgorithmHparams):
    """See :class:`ColOut`"""
    p_row: float = hp.optional(doc="Fraction of rows to drop", default=0.15)
    p_col: float = hp.optional(doc="Fraction of cols to drop", default=0.15)
    batch: bool = hp.optional(doc="Run ColOut at the batch level", default=True)

    def initialize_object(self) -> ColOut:
        return ColOut(**asdict(self))


class ColOut(Algorithm):
    """Drops a fraction of the rows and columns of an input image. If the
    fraction of rows/columns dropped isn't too large, this does not
    significantly alter the content of the image, but reduces its size
    and provides extra variability.

    Args:
        p_row: Fraction of rows to drop (drop along H).
        p_col: Fraction of columns to drop (drop along W).
        batch: Run ColOut at the batch level.
    """

    def __init__(self, p_row: float = 0.15, p_col: float = 0.15, batch: bool = True):
        if not (0 <= p_col <= 1):
            raise ValueError("p_col must be between 0 and 1")

        if not (0 <= p_row <= 1):
            raise ValueError("p_row must be between 0 and 1")

        self.hparams = ColOutHparams(p_row, p_col, batch)

    def match(self, event: Event, state: State) -> bool:
        """Apply on Event.TRAINING_START for samplewise or Event.AFTER_DATALOADER for batchwise """
        if self.hparams.batch:
            return event == Event.AFTER_DATALOADER
        else:
            return event == Event.TRAINING_START

    def _apply_sample(self, state: State) -> None:
        """Add the ColOut dataset transform to the dataloader """
        assert state.train_dataloader is not None
        dataset = state.train_dataloader.dataset

        transform = ColOutTransform(p_row=self.hparams.p_row, p_col=self.hparams.p_col)

        if hasattr(dataset, "transform"):
            add_dataset_transform(dataset, transform)
        else:
            raise ValueError(
                f"Dataset of type {type(dataset)} has no attribute 'transform'. Expected TorchVision dataset.")

    def _apply_batch(self, state: State) -> None:
        """Transform a batch of images using the ColOut augmentation """
        inputs, labels = state.batch_pair
        assert isinstance(inputs, Tensor), "Multiple Tensors not supported yet for ColOut"
        new_inputs = batch_colout(inputs, p_row=self.hparams.p_row, p_col=self.hparams.p_col)

        state.batch = (new_inputs, labels)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Applies ColOut augmentation to the state's input

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Optional[Logger], optional): the training logger. Defaults to None.
        """
        if self.hparams.batch:
            self._apply_batch(state)
        else:
            self._apply_sample(state)
