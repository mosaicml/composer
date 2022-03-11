# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Progressive Resizing classes and functions."""

from __future__ import annotations

import logging
import textwrap
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Tensor
from composer.models.loss import _check_for_index_targets

log = logging.getLogger(__name__)

_VALID_MODES = ("crop", "resize")

T_ResizeTransform = Callable[[torch.Tensor], torch.Tensor]

__all__ = ["resize_batch", "ProgressiveResizing"]


def resize_batch(input: torch.Tensor,
                 target: torch.Tensor,
                 scale_factor: float,
                 mode: str = "resize",
                 resize_targets: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize inputs and optionally outputs by cropping or interpolating.

    Args:
        input (torch.Tensor): input tensor of shape ``(N, C, H, W)``.
            Resizing will be done along dimensions H and W using the constant
            factor ``scale_factor``.
        target (torch.Tensor): output tensor of shape ``(N, H, W)`` or
            ``(N, C, H, W)`` that will also be resized if ``resize_targets``
            is ``True``,
        scale_factor (float): scaling coefficient for the height and width of the
            input/output tensor. 1.0 keeps the original size.
        mode (str, optional): type of scaling to perform. Value must be one of ``'crop'`` or
            ``'resize'``. ``'crop'`` performs a random crop, whereas ``'resize'``
            performs a nearest neighbor interpolation. Default: ``"resize"``.
        resize_targets (bool, optional): whether to resize the targets, ``y``. Default: ``False``.

    Returns:
        X_sized: resized input tensor of shape ``(N, C, H * scale_factor, W * scale_factor)``.
        y_sized: if ``resized_targets`` is ``True``, resized output tensor
            of shape ``(N, H * scale_factor, W * scale_factor)`` or  ``(N, C, H * scale_factor, W * scale_factor)``.
            Depending on the input ``y``. Otherwise returns original ``y``.

    Example:
         .. testcode::

            from composer.algorithms.progressive_resizing import resize_batch
            X_resized, y_resized = resize_batch(X_example,
                                                y_example,
                                                scale_factor=0.5,
                                                mode='resize',
                                                resize_targets=False)
    """
    # Verify dimensionalities are enough to support resizing
    assert input.dim() > 2, "Input dimensionality not large enough for resizing"
    if resize_targets is True:
        assert target.dim() > 2, "Target dimensionality not large enough for resizing"

    # Short-circuit if nothing should be done
    if scale_factor >= 1:
        return input, target

    # Prep targets for resizing if necessary
    if _check_for_index_targets(target) and resize_targets is True:
        # Add a dimension to match shape of the input and change type for resizing
        y_sized = target.float().unsqueeze(1)
    else:
        y_sized = target

    if mode.lower() == "crop" and resize_targets is False:
        # Make a crop transform for X
        resize_transform = _make_crop(tensor=input, scale_factor=scale_factor)
        X_sized, y_sized = resize_transform(input), target
    elif mode.lower() == "crop" and resize_targets is True:
        # Make a crop transform for X and y
        resize_transform, resize_y = _make_crop_pair(X=input, y=y_sized, scale_factor=scale_factor)
        X_sized, y_sized = resize_transform(input), resize_y(y_sized)
    elif mode.lower() == "resize":
        # Make a resize transform (can be used for X or y)
        resize_transform = _make_resize(scale_factor=scale_factor)
        X_sized = resize_transform(input)
        if resize_targets:
            y_sized = resize_transform(y_sized)
    else:
        raise ValueError(f"Progressive mode '{mode}' not supported.")

    # Revert targets to their original format if they were modified
    if _check_for_index_targets(target) and resize_targets is True:
        # Convert back to original format for training
        y_sized = y_sized.squeeze(dim=1).to(target.dtype)

    # Log results
    log.info(
        textwrap.dedent(f"""\
            Applied Progressive Resizing with scale_factor={scale_factor} and mode={mode}.
            Old input dimensions: (H,W)={input.shape[2], input.shape[3]}.
            New input dimensions: (H,W)={X_sized.shape[2], X_sized.shape[2]}"""))
    return X_sized, y_sized


class ProgressiveResizing(Algorithm):
    """Apply Fastai's `progressive resizing <https://\\
    github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb>`__ data
    augmentation to speed up training.

    Progressive resizing initially reduces input resolution to speed up early training.
    Throughout training, the downsampling factor is gradually increased, yielding larger inputs
    up to the original input size. A final finetuning period is then run to finetune the
    model using the full-sized inputs.

    Example:
         .. testcode::

            from composer.algorithms import ProgressiveResizing
            from composer.trainer import Trainer
            progressive_resizing_algorithm = ProgressiveResizing(
                                                mode='resize',
                                                initial_scale=1.0,
                                                finetune_fraction=0.2,
                                                resize_targets=False
                                            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[progressive_resizing_algorithm],
                optimizers=[optimizer]
            )

    Args:
        mode (str, optional): Type of scaling to perform. Value must be one of ``'crop'`` or ``'resize'``.
            ``'crop'`` performs a random crop, whereas ``'resize'`` performs a bilinear
            interpolation. Default: ``"resize"``.
        initial_scale (float, optional): Initial scale factor used to shrink the inputs. Must be a
            value in between 0 and 1. Default: ``0.5``.
        finetune_fraction (float, optional): Fraction of training to reserve for finetuning on the
            full-sized inputs. Must be a value in between 0 and 1. Default: ``0.2``.
        resize_targets (bool, optional): If True, resize targets also. Default: ``False``.
    """

    def __init__(self,
                 mode: str = 'resize',
                 initial_scale: float = .5,
                 finetune_fraction: float = .2,
                 resize_targets: bool = False):

        if mode not in _VALID_MODES:
            raise ValueError(f"mode '{mode}' is not supported. Must be one of {_VALID_MODES}")

        if not (0 <= initial_scale <= 1):
            raise ValueError(f"initial_scale must be between 0 and 1: {initial_scale}")

        if not (0 <= finetune_fraction <= 1):
            raise ValueError(f"finetune_fraction must be between 0 and 1: {finetune_fraction}")

        self.mode = mode
        self.initial_scale = initial_scale
        self.finetune_fraction = finetune_fraction
        self.resize_targets = resize_targets

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.AFTER_DATALOADER.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now
        """
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Applies ProgressiveResizing on input images.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors not supported for this method yet."

        # Calculate the current size of the inputs to use
        initial_size = self.initial_scale
        finetune_fraction = self.finetune_fraction
        scale_frac_elapsed = min([state.get_elapsed_duration().value / (1 - finetune_fraction), 1])

        # Linearly increase to full size at the start of the fine tuning period
        scale_factor = initial_size + (1 - initial_size) * scale_frac_elapsed

        new_input, new_target = resize_batch(input=input,
                                             target=target,
                                             scale_factor=scale_factor,
                                             mode=self.mode,
                                             resize_targets=self.resize_targets)
        state.batch = (new_input, new_target)

        if logger is not None:
            logger.metric_batch({
                "progressive_resizing/height": new_input.shape[2],
                "progressive_resizing/width": new_input.shape[3],
                "progressive_resizing/scale_factor": scale_factor
            })


def _make_crop(tensor: torch.Tensor, scale_factor: float) -> T_ResizeTransform:
    """Makes a random crop transform for an input image."""
    Hc = int(scale_factor * tensor.shape[2])
    Wc = int(scale_factor * tensor.shape[3])
    top = torch.randint(tensor.shape[2] - Hc, size=(1,))
    left = torch.randint(tensor.shape[3] - Wc, size=(1,))
    resize_transform = partial(transforms.functional.crop, top=top, left=left, height=Hc, width=Wc)
    return resize_transform


def _make_crop_pair(X: torch.Tensor, y: torch.Tensor,
                    scale_factor: float) -> Tuple[T_ResizeTransform, T_ResizeTransform]:
    """Makes a pair of random crops for an input image X and target tensor y such that the same region is selected from
    both."""
    # New height and width for X
    HcX = int(scale_factor * X.shape[2])
    WcX = int(scale_factor * X.shape[3])
    # New height and width for y
    Hcy = int(scale_factor * y.shape[2])
    Wcy = int(scale_factor * y.shape[3])
    # Select a corner for the crop from X
    topX = torch.randint(X.shape[2] - HcX, size=(1,))
    leftX = torch.randint(X.shape[3] - WcX, size=(1,))
    # Find the corresponding point for X
    height_ratio = y.shape[2] / X.shape[2]
    width_ratio = y.shape[3] / X.shape[3]
    topy = int(height_ratio * topX)
    lefty = int(width_ratio * leftX)
    # Make the two transforms
    resize_X = partial(transforms.functional.crop, top=topX, left=leftX, height=HcX, width=WcX)
    resize_y = partial(transforms.functional.crop, top=topy, left=lefty, height=Hcy, width=Wcy)
    return resize_X, resize_y


def _make_resize(scale_factor: float) -> T_ResizeTransform:
    """Makes a nearest-neighbor interpolation transform at the specified scale factor."""
    resize_transform = partial(F.interpolate, scale_factor=scale_factor, mode='nearest')
    return resize_transform
