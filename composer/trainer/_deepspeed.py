# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the `DeepSpeed <https://www.deepspeed.ai>`_ integration with Composer."""

import copy
import warnings
from typing import Any, Dict, cast

import torch
import torch.utils.data

from composer.core import Precision, State
from composer.core.types import Batch
from composer.utils import dist
from composer.utils.iter_helpers import map_collection

__all__ = ['_fix_batch_precision_for_deepspeed', '_parse_deepspeed_config']


def _add_batch_config(config: Dict[str, Any], state: State):
    if state.dataloader is None:
        raise ValueError(
            'When using DeepSpeed, the `train_dataloader` must be specified when constructing the Trainer.')

    grad_accum = state.grad_accum

    if isinstance(state.dataloader, torch.utils.data.DataLoader):
        if state.dataloader.batch_size is None:
            raise RuntimeError('DeepSpeed requires a dataloader with a known batch size.')

        if state.dataloader.batch_size % state.grad_accum != 0:
            # DeepSpeed will throw an error in this configuration.
            raise ValueError('The Mosaic trainer has been configured to use batch size='
                             f'{state.dataloader.batch_size}, but this is not divisible by the '
                             f'grad accum={state.grad_accum}. This is unsupported when using DeepSpeed.')

        train_batch_size = state.dataloader.batch_size * dist.get_world_size()
        # Per the check at the start of this function, the following division is always clean.
        per_gpu_microbatch_size = state.dataloader.batch_size // grad_accum

        if 'train_batch_size' in config:
            ds_train_batch_size = config['train_batch_size']
            if ds_train_batch_size != train_batch_size:
                raise ValueError(f'Provided DeepSpeed configuration specifies batch size={ds_train_batch_size}, '
                                 f'but the Mosaic trainer has been configured with batch size={train_batch_size}.')

        if 'train_micro_batch_size_per_gpu' in config:
            ds_per_gpu_microbatch_size = config['train_micro_batch_size_per_gpu']
            if ds_per_gpu_microbatch_size != per_gpu_microbatch_size:
                raise ValueError('Provided DeepSpeed configuration specifies per-GPU microbatch size='
                                 f'{ds_per_gpu_microbatch_size}, but the Mosaic trainer has been '
                                 f'configured with per-GPU microbatch size={per_gpu_microbatch_size}.')

        config['train_batch_size'] = train_batch_size
        config['train_micro_batch_size_per_gpu'] = per_gpu_microbatch_size

    if 'gradient_accumulation_steps' in config:
        ds_grad_accum = config['gradient_accumulation_steps']
        if ds_grad_accum != grad_accum:
            raise ValueError((f'Provided DeepSpeed configuration specifies grad accum={ds_grad_accum}, '
                              f'but the Mosaic trainer has been configured with grad accum={grad_accum}.'))

    config['gradient_accumulation_steps'] = grad_accum


def _ensure_no_optim_in_config(config: Dict[str, Any]):
    if 'optimizer' in config:
        raise ValueError(('The DeepSpeed configuration specifies an optimizer, but the Mosaic '
                          'trainer will override this setting.'))

    if 'scheduler' in config:
        raise ValueError(('The DeepSpeed configuration specifies a scheduler, but the Mosaic '
                          'trainer will override this setting.'))


def _add_precision_config(config: Dict[str, Any], state: State):
    precision = state.precision

    ds_precision = None
    if 'fp16' in config and 'enabled' in config['fp16'] and config['fp16']['enabled']:
        ds_precision = Precision.FP16
    if 'bf16' in config and 'enabled' in config['bf16'] and config['bf16']['enabled']:
        raise ValueError(('DeepSpeed is configured to use BFLOAT16, but this is unsupported by the '
                          'Mosaic trainer.'))
    if 'amp' in config and 'enabled' in config['amp'] and config['amp']['enabled']:
        raise ValueError(('DeepSpeed is configured to use Apex AMP, but this is unsupported by the '
                          'Mosaic trainer.'))

    if ds_precision is not None and ds_precision != precision:
        raise ValueError((f'Provided DeepSpeed configuration specifies precision={ds_precision}, '
                          f'but the Mosaic trainer has been configured with precision={precision}.'))

    if precision == Precision.FP16:
        if 'fp16' not in config:
            config['fp16'] = cast(Dict[str, Any], {'enabled': True})
        fp16_config = config['fp16']
        assert isinstance(fp16_config, dict)


def _parse_deepspeed_config(
    config: Dict[str, Any],
    state: State,
) -> Dict[str, Any]:
    """Parses the provided DeepSpeed config for compatibility with the Mosaic trainer.

    Broadly speaking, this function does three things.

    1. Check for settings that are unsupported, like DeepSpeed optimizers.

    2. Check for inconsistencies between Mosaic trainer config and DeepSpeed config.

    3. Use Mosaic trainer config to fill in some defaults for DeepSpeed config.

    Args:
        config (Dict[str, Any]): The DeepSpeed config to use. Must follow the format specified
            in `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_.
        state (State): The state of the trainer.

    Returns:
        Dict[str, Any]: The DeepSpeed config updated with values from the arguments passed to the
            :class:`.Trainer`.

    Raises:
        ValueError: If any of the values in the DeepSpeed config conflict with arguments passed
            to the trainer.
        RuntimeError: If the batch size of the train dataloader in the provided state is not set.
    """
    new_config = copy.deepcopy(config)
    _add_batch_config(new_config, state)
    _ensure_no_optim_in_config(new_config)
    _add_precision_config(new_config, state)
    if 'zero_allow_untested_optimizer' in new_config and not new_config['zero_allow_untested_optimizer']:
        warnings.warn(('Provided DeepSpeed configuration specifies zero_allow_untested_optimizer=False. '
                       'This causes DeepSpeed to reject certain Mosaic optimizers that are known to '
                       'work well with DeepSpeed.'))

    new_config['zero_allow_untested_optimizer'] = True
    return new_config


def _convert_fp32_tensor_to_fp16(tensor: torch.Tensor):
    if tensor.dtype == torch.float32:
        return tensor.half()
    return tensor


def _fix_batch_precision_for_deepspeed(batch: Batch, precision: Precision) -> Batch:
    """Ensures that a batch is properly formatted for DeepSpeed FP16, if active.

    .. note:: Just because the precision is set to FP16 doesn't mean the entire batch can
              be FP16 too. For example, integer tensors are common in inputs and outputs of
              various models, and these must not be converted. The assumption here is
              that a tensor should only be converted to FP16 if it was given in FP32.

    Args:
        batch (Batch): The batch of data to adjust the precision for.
        precision (Precision): The precision to use.

    Returns:
        Batch: The batch with it's precision adjusted to the specified precision.
    """
    if precision != Precision.FP16:
        return batch

    return map_collection(batch, _convert_fp32_tensor_to_fp16)  # type: ignore
