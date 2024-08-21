# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor train and eval images."""
from typing import Any, Callable, Sequence, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loss.utils import infer_target_type

__all__ = ['ImageVisualizer']


class ImageVisualizer(Callback):
    """Logs image inputs and optionally outputs.

    This callback triggers at a user defined interval, and logs a sample of input (optionally also segmentation masks)
    images under the ``Images/Train`` and ``Image/Eval`` keys.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import ImageVisualizer
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[ImageVisualizer()],
            ... )

    The images are logged by the :class:`.Logger` to the following key(s) as described
    below.

    +---------------------------------------------+---------------------------------------+
    | Key                                         | Logged data                           |
    +=============================================+=======================================+
    | ``Images/Train``                            |   Sampled examples of train images    |
    +---------------------------------------------+---------------------------------------+
    | ``Images/Eval``                             |   Sampled examples of eval images     |
    +---------------------------------------------+---------------------------------------+

        .. note::
            This callback only works with wandb and Neptune logging for now.

    Args:
        interval (int | str | Time, optional): Time string specifying how often to log train images. For example, ``interval='1ep'``
            means images are logged once every epoch, while ``interval='100ba'`` means images are logged once every 100
            batches. Eval images are logged once at the start of each eval. Default: ``"100ba"``.
        mode (str, optional): How to log the image labels. Valid values are ``"input"`` (input only)
            and "segmentation" which also saves segmentation masks. Default: ``"input"``.
        num_images (int, optional): Number of images to log. Should be less than or equal to than the microbatch size.
            If there are not enough images in the microbatch, all the images in the microbatch will be logged.
            Default: ``8``.
        channels_last (bool, optional): Whether the image channel dimension is the last dimension. Default: ``False``.
        input_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first
            element is the input. Default: ``0``.
        target_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second
            element is the target. Default: ``1``.
    """

    def __init__(
        self,
        interval: Union[int, str, Time] = '100ba',
        mode: str = 'input',
        num_images: int = 8,
        channels_last: bool = False,
        input_key: Union[str, int, tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, tuple[Callable, Callable], Any] = 1,
    ):
        self.mode = mode
        self.num_images = num_images
        self.channels_last = channels_last
        self.input_key = input_key
        self.target_key = target_key

        # Check that the output mode is valid
        if self.mode.lower() not in ['input', 'segmentation']:
            raise ValueError(f'Invalid mode: {mode}')

        # Check that the interval timestring is parsable and convert into time object
        self.interval = Time.from_input(interval, TimeUnit.BATCH)

        # Verify that the interval has supported units
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(
                f'Invalid time unit for parameter interval: '
                f'{self.interval.unit}',
            )

        self.last_train_time_value_logged = -1

    def _log_inputs(self, state: State, logger: Logger, data_name: str):
        inputs = state.batch_get_item(key=self.input_key)
        # Verify inputs is a valid shape for conversion to an image
        if _check_for_image_format(inputs):
            inputs = _make_input_images(inputs, self.num_images)
            logger.log_images(inputs, name=data_name, use_table=True, channels_last=self.channels_last)

    def _log_segmented_inputs(self, state: State, logger: Logger, data_name: str):
        inputs = state.batch_get_item(key=self.input_key)
        targets = state.batch_get_item(key=self.target_key)
        outputs = state.outputs
        assert isinstance(outputs, torch.Tensor)

        images, masks = _make_segmentation_images(inputs, targets, outputs, self.num_images, self.channels_last)
        logger.log_images(images, masks=masks, name=data_name, channels_last=self.channels_last, use_table=True)

    def before_forward(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value
        if self.mode.lower(
        ) == 'input' and current_time_value % self.interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self._log_inputs(state, logger, 'Images/Train')

    def eval_before_forward(self, state: State, logger: Logger):
        if self.mode.lower() == 'input' and state.eval_timestamp.get(TimeUnit.BATCH).value == 0:
            self._log_inputs(state, logger, 'Images/Eval')

    def before_loss(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value
        if self.mode.lower(
        ) == 'segmentation' and current_time_value % self.interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self._log_segmented_inputs(state, logger, 'Images/Train')

    def eval_after_forward(self, state: State, logger: Logger):
        if self.mode.lower() == 'segmentation' and state.eval_timestamp.get(TimeUnit.BATCH).value == 0:
            self._log_segmented_inputs(state, logger, 'Images/Eval')


def _make_input_images(inputs: torch.Tensor, num_images: int):
    if inputs.shape[0] < num_images:
        num_images = inputs.shape[0]
    images = inputs[0:num_images].data.cpu().numpy()
    return images


def _make_segmentation_images(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    num_images: int,
    channels_last: bool = False,
):
    if isinstance(outputs, Sequence):
        outputs = torch.stack(list(outputs))
    if min([inputs.shape[0], targets.shape[0], outputs.shape[0]]) < num_images:
        num_images = min([inputs.shape[0], targets.shape[0], outputs.shape[0]])

    images = inputs[0:num_images].data.cpu().numpy()

    targets = targets[0:num_images]
    outputs = outputs[0:num_images]
    # Ensure the targets are in the expected format
    if infer_target_type(outputs, targets) == 'one_hot':
        if channels_last:
            targets = targets.argmax(dim=-1).data.cpu().numpy()  # type: ignore
        else:
            targets = targets.argmax(dim=1).data.cpu().numpy()  # type: ignore
    else:
        targets = targets.data.cpu().numpy()  # type: ignore
    # Convert the outputs to the expected format
    if channels_last:
        num_classes = outputs.shape[-1]
        outputs = outputs.argmax(dim=-1).cpu().numpy()  # type: ignore
    else:
        num_classes = outputs.shape[1]
        outputs = outputs.argmax(dim=1).cpu().numpy()  # type: ignore
    # Adjust targets such that negative values are mapped to one higher than the maximum class
    targets[targets < 0] = num_classes

    return images, {'prediction': outputs, 'ground_truth': targets}


def _check_for_image_format(data: torch.Tensor) -> bool:
    return data.ndim in [3, 4] and data.numel() > data.shape[0]
