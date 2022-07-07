# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor train and eval images."""
from typing import Any, Callable, Tuple, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['ImageMonitor']


class ImageMonitor(Callback):
    """Logs image inputs and optionally outputs.

    This callback triggers at a user defined interval, and logs a sample of input (optionally also segmentation masks)
    images under the ``Images/Train`` and ``Image/Eval`` keys.

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
    | ``Images/Train``                            |   Sampled examples of train images    |
    +---------------------------------------------+---------------------------------------+
    | ``Images/Eval``                             |   Sampled examples of eval images     |
    +---------------------------------------------+---------------------------------------+

    Args:
        interval (str, optional): Time string specifying how often to log train images. For example, ``interval='1ep'``
            means images are logged once every epoch, while ``interval='100ba'`` means images are logged once every 100
            batches. Eval images are logged once at the start of each eval. Default: ``"100ba"``.
        mode (str, optional): How to log the image labels. Valid values are ``"input"`` (input only)
            and "segmentation" which also saves segmentation masks. Default: ``"input"``.
        num_images (int, optional): Number of images to log. Should be less than or equal to than the microbatch size.
            If there are not enough images in the microbatch, all the images in the microbatch will be logged.
            Default: ``8``.
        shift_targets (bool): For use when ``mode="segmentation"``. Optionally shifts the target classes up by one to
            allow for ignored/background classes with negative values. If ``shift_targets=True``, all class ids will be
            incremented by 1, and any pixels with negative target indices will be assigned a target index of zero.
            Default: ``False``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first
            element is the input. Default: ``0``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second
            element is the target. Default: ``1``.
    """

    def __init__(self,
                 interval: str = '100ba',
                 mode: str = 'input',
                 num_images: int = 8,
                 shift_targets: bool = False,
                 input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
                 target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1):
        self.interval = interval
        self.mode = mode
        self.num_images = num_images
        self.shift_targets = shift_targets
        self.input_key = input_key
        self.target_key = target_key

        try:
            import wandb
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='wandb',
                                                conda_package='wandb',
                                                conda_channel='conda-forge') from e
        del wandb  # unused

        # Check that the output mode is valid
        if self.mode.lower() not in ['input', 'segmentation']:
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

    def _log_inputs(self, state: State, logger: Logger, key: str):
        inputs = state.batch_get_item(key=self.input_key)
        if not isinstance(inputs, torch.Tensor):
            raise NotImplementedError('Multiple input tensors not supported yet')
        # Verify inputs is a valid shape for conversion to an image
        if _check_for_image_format(inputs):
            table = _make_input_images(inputs, self.num_images)
            logger.data_batch({key: table})

    def _log_segmented_inputs(self, state: State, logger: Logger, key: str):
        inputs = state.batch_get_item(key=self.input_key)
        targets = state.batch_get_item(key=self.target_key)
        outputs = state.outputs

        if not isinstance(inputs, torch.Tensor):
            raise NotImplementedError('Multiple input tensors not supported yet')
        if not isinstance(targets, torch.Tensor):
            raise NotImplementedError('Multiple target tensors not supported yet')
        if not isinstance(outputs, torch.Tensor):
            raise NotImplementedError('Multiple output tensors not supported yet')

        table = _make_segmentation_images(inputs, targets, outputs, self.num_images, self.shift_targets)
        logger.data_batch({key: table})

    def before_forward(self, state: State, logger: Logger):
        assert isinstance(self.interval, Time)
        if self.mode.lower() == 'input' and state.timestamp.get(self.interval.unit).value % self.interval.value == 0:
            self._log_inputs(state, logger, 'Images/Train')

    def eval_before_forward(self, state: State, logger: Logger):
        assert isinstance(self.interval, Time)
        if self.mode.lower() == 'input' and state.eval_timestamp.get(TimeUnit.BATCH).value == 0:
            self._log_inputs(state, logger, 'Images/Eval')

    def before_loss(self, state: State, logger: Logger):
        assert isinstance(self.interval, Time)
        if self.mode.lower() == 'segmentation' and state.timestamp.get(
                self.interval.unit).value % self.interval.value == 0:
            self._log_segmented_inputs(state, logger, 'Images/Train')

    def eval_after_forward(self, state: State, logger: Logger):
        assert isinstance(self.interval, Time)
        if self.mode.lower() == 'segmentation' and state.eval_timestamp.get(TimeUnit.BATCH).value == 0:
            self._log_segmented_inputs(state, logger, 'Images/Eval')


def _make_input_images(inputs: torch.Tensor, num_images: int):
    import wandb
    if inputs.shape[0] < num_images:
        num_images = inputs.shape[0]
    images = inputs[0:num_images].data.cpu().permute(0, 2, 3, 1).numpy()
    table = wandb.Table(columns=['Image'])
    for image in images:
        img = wandb.Image(image)
        table.add_data(img)
    return table


def _make_segmentation_images(inputs: torch.Tensor,
                              targets: torch.Tensor,
                              outputs: torch.Tensor,
                              num_images: int,
                              shift_targets: bool = False):
    import wandb
    if min([inputs.shape[0], targets.shape[0], outputs.shape[0]]) < num_images:
        num_images = min([inputs.shape[0], targets.shape[0], outputs.shape[0]])
    images = inputs[0:num_images].data.cpu().permute(0, 2, 3, 1).numpy()
    targets = targets[0:num_images].data.cpu().numpy()
    outputs = outputs[0:num_images]
    # Convert outputs to segmentation masks. Assume channels are first dim
    outputs = outputs[0:num_images]
    outputs = outputs.argmax(dim=1).cpu().numpy()
    # Shift targets such that negative values are mapped to 0
    if shift_targets:
        targets += 1
        targets[targets < 0] = 0
        outputs += 1
    table = wandb.Table(columns=['Image'])
    for image, target, prediction in zip(images, targets, outputs):
        mask = {'ground truth': {'mask_data': target}, 'prediction': {'mask_data': prediction}}
        img_mask_pair = wandb.Image(image, masks=mask)
        table.add_data(img_mask_pair)
    return table


def _check_for_image_format(data: torch.Tensor) -> bool:
    return data.ndim in [3, 4] and data.numel() > data.shape[0]
