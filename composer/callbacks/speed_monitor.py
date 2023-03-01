# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Union

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.models.base import ComposerModel
from composer.utils import dist

__all__ = ['SpeedMonitor']

GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    'h100-sxm': {
        'fp64': 67e12,
        'fp32': 67e12,
        'tf32': 989e12 / 2,
        'fp16': 1.979e15 / 2,
        'amp_fp16': 1.979e15 / 2,
        'bf16': 1.979e15 / 2,
        'amp_bf16': 1.979e15 / 2,
        'fp8': 3.958e15 / 2,
        'amp_fp8': 3.958e15 / 2,
        'int8': 3.958e15 / 2,
    },
    'h100-pcie': {
        'fp64': 51e12,
        'fp32': 51e12,
        'tf32': 756e12 / 2,
        'fp16': 1.513e15 / 2,
        'amp_fp16': 1.513e15 / 2,
        'bf16': 1.513e15 / 2,
        'amp_bf16': 1.513e15 / 2,
        'fp8': 3.026e15 / 2,
        'amp_fp8': 3.026e15 / 2,
        'int8': 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    'a100': {
        'fp64': 19.5e12,
        'fp32': 19.5e12,
        'tf32': 156e12,
        'fp16': 312e12,
        'amp_fp16': 312e12,
        'bf16': 312e12,
        'amp_bf16': 312e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    'v100-sxm': {
        'fp64': 7.8e12,
        'fp32': 15.7e12,
        'fp16': 125e12,
        'amp_fp16': 125e12,
    },
    'v100-pcie': {
        'fp64': 7e12,
        'fp32': 14e12,
        'fp16': 112e12,
        'amp_fp16': 112e12,
    },
    'v100s-pcie': {
        'fp64': 8.2e12,
        'fp32': 16.4e12,
        'fp16': 130e12,
        'amp_fp16': 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    't4': {
        'fp32': 8.1e12,
        'fp16': 65e12,
        'amp_fp16': 65e12,
        'int8': 130e12,
        'int4': 260e12,
    },
}


def get_gpu_flops_available(state: State):
    gpu_flops_available = None

    # Return 0 if no CUDA device (e.g., when running with CPU only)
    if not torch.cuda.is_available():
        return 0

    # torch.cuda.get_device_name() ex output: 'NVIDIA A100-SXM4-40GB'
    device_name = torch.cuda.get_device_name().lower()
    if 'h100-sxm' in device_name:
        device_name = 'h100-sxm'
    elif 'h100-pcie' in device_name:
        device_name = 'h100-pcie'
    elif 'a100' in device_name:
        device_name = 'a100'
    elif 'v100-sxm' in device_name:
        device_name = 'v100-sxm'
    elif 'v100-pcie' in device_name:
        device_name = 'v100-pcie'
    elif 't4' in device_name:
        device_name = 't4'
    else:
        device_name = None

    if device_name is not None:
        try:
            gpu_flops_available = int(GPU_AVAILABLE_FLOPS[device_name][state.precision.value])
        except:
            gpu_flops_available = None

    if gpu_flops_available is None:
        warnings.warn(
            f'gpu_flop count not found for {device_name} with precision: {state.precision.value}; ' +\
            f'MFU cannot be calculated and reported. gpu_flops_available can be manually' +\
            f'overridden by setting gpu_flops_available in SpeedMonitor.'
        )
        # Setting to 0 will disable MFU computation and prevent
        # the speed monitor from running this helper every batch
        gpu_flops_available = 0

    return gpu_flops_available


class SpeedMonitor(Callback):
    """Logs the training throughput and utilization.

    The training throughput is logged on the :attr:`.Event.BATCH_END` event once we have reached
    the `window_size` threshold. If a model has `flops_per_batch` attribute, then flops per second
    is also logged. If running on a known GPU type or if `gpu_flops_available` is set, then MFU is
    also logged. All metrics are also logged as per device by dividing by world size.

    To compute `flops_per_sec`, the model attribute `flops_per_batch` should be set to a callable
    which accepts a batch and returns the number of flops for that batch. Typically, this should
    be flops per sample times the batch size unless pad tokens are used.

    The wall clock time is logged on every :attr:`.Event.BATCH_END` event.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import SpeedMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[SpeedMonitor(window_size=100)],
            ... )

    The training throughput is logged by the :class:`.Logger` to the following keys as
    described below.

    +-------------------------------------+-----------------------------------------------------------+
    | Key                                 | Logged data                                               |
    +=====================================+===========================================================+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/batches_per_sec`        | batches) of the number of batches processed per second    |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/samples_per_sec`        | batches) of the number of samples processed per second    |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/tokens_per_sec`         | batches) of the number of tokens processed per second.    |
    |                                     | Only logged when dataloader.dataset has `max_seq_len`.    |
    |                                     | This may include padding depending on dataset             |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Estimates flops by `flops_per_batch * batches_per_sec`    |
    | `throughput/flops_per_sec`          | if model has attribute `flops_per_batch`                  |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/batches_per_sec` | `throughput/batches_per_sec` divided by world size        |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/samples_per_sec` | `throughput/samples_per_sec` divided by world size        |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/tokens_per_sec` divided by world size. Only   |
    | `throughput/device/tokens_per_sec`  | logged when dataloader.dataset has `max_seq_len`. This    |
    |                                     | may include pad tokens depending on dataset               |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/flops_per_sec` divided by world size. Only    |
    | `throughput/device/flops_per_sec`   | logged when model has attribute `flops_per_batch`         |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/device/flops_per_sec` divided by world size.  |
    | `throughput/device/mfu`             | Only logged when model has attribute `flops_per_batch`    |
    |                                     | and `gpu_flops_available`, which can be passed as an      |
    |                                     | argument if not automatically determined by SpeedMonitor  |
    +-------------------------------------+-----------------------------------------------------------+
    | `wall_clock/train`                  | Total elapsed training time                               |
    +-------------------------------------+-----------------------------------------------------------+
    | `wall_clock/val`                    | Total elapsed validation time                             |
    +-------------------------------------+-----------------------------------------------------------+
    | `wall_clock/total`                  | Total elapsed time (wall_clock/train + wall_clock/val)    |
    +-------------------------------------+-----------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
        gpu_flops_available (float, optional): Number of flops available on the GPU. If not set,
            SpeedMonitor will attempt to determine this automatically. Defaults to None.
        time_unit (str, optional): Time unit to use for `wall_clock` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.
    """

    def __init__(
        self,
        window_size: int = 100,
        gpu_flops_available: Optional[Union[float, int]] = None,
        time_unit: str = 'hours',
    ):
        # Track the batch num samples and wct to compute throughput over a window of batches
        self.history_samples: Deque[int] = deque(maxlen=window_size + 1)
        self.history_wct: Deque[float] = deque(maxlen=window_size + 1)
        self.history_flops: Deque[float] = deque(maxlen=window_size + 1)

        self.gpu_flops_available = gpu_flops_available

        self.divider = 1
        if time_unit == 'seconds':
            self.divider = 1
        elif time_unit == 'minutes':
            self.divider = 60
        elif time_unit == 'hours':
            self.divider = 60 * 60
        elif time_unit == 'days':
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".')

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            'total_eval_wct': self.total_eval_wct,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.total_eval_wct = state['total_eval_wct']

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if self.gpu_flops_available is None:
            self.gpu_flops_available = get_gpu_flops_available(state)

    def batch_end(self, state: State, logger: Logger):
        # Add the new element
        self.history_samples.append(state.timestamp.sample.value)
        self.history_wct.append(state.timestamp.total_wct.total_seconds())

        # Log the throughput
        if len(self.history_wct) == self.history_wct.maxlen:
            world_size = dist.get_world_size()
            elapsed_batches = len(self.history_samples) - 1
            elapsed_samples = int(self.history_samples[-1]) - int(self.history_samples[0])
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            batches_per_sec = elapsed_batches / elapsed_wct
            samples_per_sec = elapsed_samples / elapsed_wct
            dev_batches_per_sec = batches_per_sec / world_size
            dev_samples_per_sec = samples_per_sec / world_size
            logger.log_metrics({'throughput/batches_per_sec': batches_per_sec})
            logger.log_metrics({'throughput/samples_per_sec': samples_per_sec})
            logger.log_metrics({'throughput/device/batches_per_sec': dev_batches_per_sec})
            logger.log_metrics({'throughput/device/samples_per_sec': dev_samples_per_sec})

            # Compute token stats if dataloader.dataset has max_seq_len. Assumes no padding.
            try:
                max_seq_len = state.dataloader.dataset.max_seq_len  # type: ignore
                # Only applicable to seq data / models
                logger.log_metrics({'throughput/tokens_per_sec': samples_per_sec * max_seq_len})
                logger.log_metrics({'throughput/device/tokens_per_sec': dev_samples_per_sec * max_seq_len})
            except AttributeError:
                pass

        # Compute flops stats if model has flops_per_batch
        composer_model = state.model
        if not isinstance(composer_model, ComposerModel):
            composer_model = composer_model.module  # Pass through DDP wrapping
        if hasattr(composer_model, 'flops_per_batch'):
            model_flops_per_batch = composer_model.flops_per_batch  # type: ignore
            if not isinstance(model_flops_per_batch, Callable):
                raise TypeError('flops_per_batch must a callable accepting a batch and '
                                f'returning an int or float. Instead, got {type(model_flops_per_batch)}.')
            device_flops_per_batch = model_flops_per_batch(state.batch)

            # Sum flops across all ranks since each rank computes the flops for its own batch
            flops_per_batch_tensor = state.device.tensor_to_device(
                torch.tensor(device_flops_per_batch, dtype=torch.float))
            dist.all_reduce(flops_per_batch_tensor, reduce_operation='SUM')
            flops_per_batch = flops_per_batch_tensor.item()

            self.history_flops.append(flops_per_batch)

        # Log the flops throughput
        if len(self.history_flops) == self.history_flops.maxlen:
            world_size = dist.get_world_size()
            elapsed_flops = sum(self.history_flops) - self.history_flops[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            flops_per_sec = elapsed_flops / elapsed_wct
            device_flops_per_sec = flops_per_sec / world_size
            logger.log_metrics({'throughput/flops_per_sec': flops_per_sec})
            logger.log_metrics({'throughput/device/flops_per_sec': device_flops_per_sec})
            if self.gpu_flops_available:
                mfu = device_flops_per_sec / self.gpu_flops_available
                logger.log_metrics({'throughput/device/mfu': mfu})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        train_wct = state.timestamp.total_wct.total_seconds()
        logger.log_metrics({
            'wall_clock/train': train_wct / self.divider,
            'wall_clock/val': self.total_eval_wct / self.divider,
            'wall_clock/total': (train_wct + self.total_eval_wct) / self.divider,
        })

    def eval_end(self, state: State, logger: Logger):
        del logger  # unused
        self.total_eval_wct += state.eval_timestamp.total_wct.total_seconds()
