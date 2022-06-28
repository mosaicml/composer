# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor image inputs and optionally outputs."""
from math import floor, sqrt
from typing import Any, Callable, Optional, Tuple, Union
import wandb

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
        output_mode (str, optional): How to log the output images. Valid values are ``"None"`` (no outputs)
            and "segmentation" which saves segmentation masks. Default: ``"None"``.
        num_images (int, optional): Number of images to log. Must be a perfect square that is no bigger than the
            microbatch size. Default: ``8``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
    """

    def __init__(self,
                 interval: str = '100ba',
                 output_mode: Optional[str] = 'None',
                 num_images: int = 8,
                 input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0):
        self.interval = interval
        self.output_mode = output_mode
        self.num_images = num_images
        self.input_key = input_key

        # Check that the output mode is valid
        if self.output_mode not in ['None', 'segmentation']:
            raise ValueError(f'Invalid output mode: {output_mode}')

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
        input = state.batch_get_item(key=self.input_key)
        images = make_grid(input[0:self.num_images], nrow=self.nrow, normalize=True)
        images = wandb.Image(images)
        logger.data_batch({'Images/Inputs': images})
