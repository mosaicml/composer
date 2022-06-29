# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor image inputs and optionally outputs."""
from math import floor, sqrt
from typing import Any, Callable, Optional, Tuple, Union

import wandb
from torchvision.utils import make_grid

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger

__all__ = ['ImageMonitor']


class ImageMonitor(Callback):
    """Logs image inputs and optionally outputs.

    This callback triggers at a user defined interval, and logs a sample of input (optionally also output) images under
    the ``Images/Inputs`` key (optionally also ``Images/Outputs`` key).

    Example:
    .. doctest::

        >>> from composer import Trainer
        >>> from composer.callbacks import ImageMonitor
        >>> # constructing trainer object with this callback
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     eval_dataloader=eval_dataloader,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[ImageMonitor()],
        ... )

    The images are logged by the :class:`.Logger` to the following key(s) as described
    below.

    +---------------------------------------------+---------------------------------------+
    | Key                                         | Logged data                           |
    +=============================================+=======================================+
    | ``Images/Inputs``                           |   Sampled examples of input images    |
    +---------------------------------------------+---------------------------------------+
    | ``Images/Outputs``                          |   Sampled examples of output images   |
    +---------------------------------------------+---------------------------------------+

    Args:
        interval (str, optional): Time string specifying how often to log images. For example, ``interval='1ep'`` means
            images are logged once every epoch, while ``interval='100ba'`` means images are logged once every 100
            batches. Default: ``"100ba"``.
        mode (str, optional): How to log the image labels. Valid values are ``"input"`` (input only)
            and "segmentation" which saves segmentation masks. Default: ``"input"``.
        num_images (int, optional): Number of images to log. Must be a perfect square that is no bigger than the
            microbatch size. Default: ``8``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.
    """

    def __init__(self,
                 interval: str = '100ba',
                 mode: str = 'input',
                 num_images: int = 8,
                 input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
                 target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1):
        self.interval = interval
        self.mode = mode
        self.num_images = num_images
        self.input_key = input_key
        self.target_key = target_key

        # Check that the output mode is valid
        if self.mode not in ['input', 'segmentation']:
            raise ValueError(f'Invalid mode: {mode}')

        # Check that the interval timestring is parsable and convert into time object
        try:
            self.interval = Time.from_timestring(interval)
        except ValueError as error:
            raise ValueError(f'Invalid time string for parameter interval') from error

        # Verify that the interval has supported units.
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.interval.unit}')

        # Ensure that the number of images is a perfect square
        self.nrow = floor(sqrt(self.num_images))
        self.num_images = self.nrow * self.nrow

    def before_forward(self, state, logger):
        if self.mode.lower() == 'input':
            inputs = state.batch_get_item(key=self.input_key)
            targets = state.batch_get_item(key=self.target_key)

            images = make_grid(input[0:self.num_images], nrow=self.nrow, normalize=True)
            images = wandb.Image(images)
            logger.data_batch({'Images/Inputs': images})

    def after_forward(self, state, logger):
        inputs = state.batch_get_item(key=self.input_key)
        targets = state.batch_get_item(key=self.target_key)

        images = inputs[0:self.num_images].data.cpu().permute(0,2,3,1).numpy()
        targets = targets[0:self.num_images].data.cpu().numpy()
        # Shift targets so that the background class is 0
        shift = targets.min()
        targets -= shift
        # Convert outputs to segmentation masks. Assume channels are first dim
        outputs = state.outputs[0:self.num_images]
        outputs = outputs.argmax(dim=1).cpu().numpy()
        outputs -= shift

        elif self.mode.lower() == 'segmentation':
            mask_list = []
            for image, target, output in zip(images, targets, outputs):
                img_mask_pair = wandb.Image(image, masks={"ground truth": {"mask_data": target}, "prediction": {"mask_data": output}})
                mask_list.append(img_mask_pair)
            logger.data_batch({'Images/Inputs': mask_list})
