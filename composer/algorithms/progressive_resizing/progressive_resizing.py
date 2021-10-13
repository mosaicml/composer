# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import yahp as hp
from torchvision import transforms

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Tensor

log = logging.getLogger(__name__)

_VALID_MODES = ("crop", "resize")


def resize_inputs(X: torch.Tensor,
                  y: torch.Tensor,
                  scale_factor: float,
                  mode: str = "resize",
                  resize_targets: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize inputs and optionally outputs by cropping or interpolating.

    Args:
        X: input tensor of shape (N, C, H, W). Resizing will be done along
            dimensions H and W using the constant factor ``scale_factor``.
        y: output tensor of shape (N, C, H, W) that will also be resized if
            ``resize_targets`` is ``True``,
        scale_factor: scaling coefficient for the height and width of the
            input/output tensor. 1.0 keeps the original size.
        mode: type of scaling to perform. Value must be one of ``'crop'`` or
            ``'resize'``. ``'crop'`` performs a random crop, whereas ``'resize'``
            performs a bilinear interpolation.
        resize_targets: whether to resize the targets, ``y``, as well

    Returns:
        X_sized: resized input tensor of shape ``(N, C, H * scale_factor, W * scale_factor)``.
        y_sized: if ``resized_targets`` is ``True``, resized output tensor
            of shape ``(N, C, H * scale_factor, W * scale_factor)``. Otherwise
            returns original ``y``.

    """
    # Short-circuit if nothing should be done
    if scale_factor >= 1:
        return X, y

    def resize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # Reduce the size of input images, either via cropping or downsampling
        if mode.lower() == "crop":
            Hc = int(scale_factor * tensor.shape[2])
            Wc = int(scale_factor * tensor.shape[3])
            resize_transform = transforms.RandomCrop((Hc, Wc))
        elif mode.lower() == "resize":
            resize_transform = partial(F.interpolate, scale_factor=scale_factor, mode='bilinear')
        else:
            raise ValueError(f"Progressive mode '{mode}' not supported.")
        return resize_transform(tensor)

    X_sized = resize_tensor(X)
    if resize_targets:
        y_sized = resize_tensor(y)
    else:
        y_sized = y
    return X_sized, y_sized


@dataclass
class ProgressiveResizingHparams(AlgorithmHparams):
    """See :class:`ProgressiveResizing`"""

    mode: str = hp.optional(doc="Type of scaling to perform", default="resize")
    initial_scale: float = hp.optional(doc="Initial scale factor", default=0.5)
    finetune_fraction: float = hp.optional(doc="Fraction of training to reserve for finetuning on full-sized inputs",
                                           default=0.2)
    resize_targets: bool = hp.optional(doc="Also resize targets", default=False)

    def initialize_object(self) -> ProgressiveResizing:
        return ProgressiveResizing(**asdict(self))


class ProgressiveResizing(Algorithm):
    """Apply Fastai's
    `progressive resizing <https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb>`_
    data augmentation to speed up training

    Progressive resizing initially reduces input resolution to speed up early training.
    Throughout training, the downsampling factor is gradually increased, yielding larger inputs
    up to the original input size. A final finetuning period is then run to finetune the
    model using the full-sized inputs.

    Args:
        mode: Type of scaling to perform. Value must be one of ``'crop'`` or ``'resize'``.
            ``'crop'`` performs a random crop, whereas ``'resize'`` performs a bilinear
            interpolation.
        initial_scale: Initial scale factor used to shrink the inputs. Must be a
            value in between 0 and 1.
        finetune_fraction: Fraction of training to reserve for finetuning on the
            full-sized inputs. Must be a value in between 0 and 1.
        resize_targets: If True, resize targets also.
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

        self.hparams = ProgressiveResizingHparams(mode=mode,
                                                  initial_scale=initial_scale,
                                                  finetune_fraction=finetune_fraction,
                                                  resize_targets=resize_targets)

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.AFTER_DATALOADER
        
        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now
        """
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Applies ProgressiveResizing on input images

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors not supported for this method yet."

        # Calculate the current size of the inputs to use
        initial_size = self.hparams.initial_scale
        finetune_fraction = self.hparams.finetune_fraction
        scale_frac_elapsed = min([(state.epoch / state.max_epochs) / (1 - finetune_fraction), 1])

        # Linearly increase to full size at the start of the fine tuning period
        scale_factor = initial_size + (1 - initial_size) * scale_frac_elapsed

        new_input, new_target = resize_inputs(X=input,
                                              y=target,
                                              scale_factor=scale_factor,
                                              mode=self.hparams.mode,
                                              resize_targets=self.hparams.resize_targets)
        state.batch = (new_input, new_target)
