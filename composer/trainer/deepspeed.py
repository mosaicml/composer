# Copyright 2021 MosaicML. All Rights Reserved.

import copy
import warnings
from typing import Any, Dict, Optional, cast

import torch

from composer.core import State
from composer.core.types import Batch, Precision, Tensor
from composer.utils import dist
from composer.utils.iter_helpers import map_collection


def parse_batch_settings(config: Dict[str, Any], state: State):
    if state.train_dataloader.batch_size is None:
        raise RuntimeError("DeepSpeed requires a dataloader with a known batch size.")

    if state.train_dataloader.batch_size % state.grad_accum != 0:
        # DeepSpeed will throw an error in this configuration.
        raise ValueError("The Mosaic trainer has been configured to use batch size="
                         f"{state.train_dataloader.batch_size}, but this is not divisible by the "
                         f"grad accum={state.grad_accum}. This is unsupported when using DeepSpeed.")

    train_batch_size = state.train_dataloader.batch_size * dist.get_world_size()
    grad_accum = state.grad_accum
    # Per the check at the start of this function, the following division is always clean.
    per_gpu_microbatch_size = state.train_dataloader.batch_size // state.grad_accum

    if "train_batch_size" in config:
        ds_train_batch_size = config["train_batch_size"]
        if ds_train_batch_size != train_batch_size:
            raise ValueError(f"Provided DeepSpeed configuration specifies batch size={ds_train_batch_size}, "
                             f"but the Mosaic trainer has been configured with batch size={train_batch_size}.")

    if "gradient_accumulation_steps" in config:
        ds_grad_accum = config["gradient_accumulation_steps"]
        if ds_grad_accum != grad_accum:
            raise ValueError(f"Provided DeepSpeed configuration specifies grad accum={ds_grad_accum}, "
                             f"but the Mosaic trainer has been configured with grad accum={grad_accum}.")

    if "train_micro_batch_size_per_gpu" in config:
        ds_per_gpu_microbatch_size = config["train_micro_batch_size_per_gpu"]
        if ds_per_gpu_microbatch_size != per_gpu_microbatch_size:
            raise ValueError("Provided DeepSpeed configuration specifies per-GPU microbatch size="
                             f"{ds_per_gpu_microbatch_size}, but the Mosaic trainer has been "
                             f"configured with per-GPU microbatch size={per_gpu_microbatch_size}.")

    config["train_batch_size"] = train_batch_size
    config["gradient_accumulation_steps"] = grad_accum
    config["train_micro_batch_size_per_gpu"] = per_gpu_microbatch_size


def parse_unsupported_settings(config: Dict[str, Any]):
    if "optimizer" in config:
        raise ValueError("The DeepSpeed configuration specifies an optimizer, but the Mosaic "
                         "trainer will override this setting.")

    if "scheduler" in config:
        raise ValueError("The DeepSpeed configuration specifies a scheduler, but the Mosaic "
                         "trainer will override this setting.")


def parse_precision_settings(config: Dict[str, Any], state: State):
    precision = state.precision

    ds_precision = None
    if "fp16" in config and "enabled" in config["fp16"] and config["fp16"]["enabled"]:
        ds_precision = Precision.FP16
    if "bf16" in config and "enabled" in config["bf16"] and config["bf16"]["enabled"]:
        raise ValueError("DeepSpeed is configured to use BFLOAT16, but this is unsupported by the "
                         "Mosaic trainer.")
    if "amp" in config and "enabled" in config["amp"] and config["amp"]["enabled"]:
        raise ValueError("DeepSpeed is configured to use Apex AMP, but this is unsupported by the "
                         "Mosaic trainer.")

    if ds_precision is not None and ds_precision != precision:
        raise ValueError(f"Provided DeepSpeed configuration specifies precision={ds_precision}, "
                         f"but the Mosaic trainer has been configured with precision={precision}.")

    if precision == Precision.FP16:
        if "fp16" not in config:
            config["fp16"] = cast({"enabled": True}, Dict[str, Any])
        fp16_config = config["fp16"]
        assert isinstance(fp16_config, dict)

        # For equivalence with the non-DeepSpeed defaults of the Mosaic trainer.
        fp16_config.setdefault("initial_scale_power", 16)
        fp16_config.setdefault("loss_scale_window", 2000)


def parse_misc_settings(config: Dict[str, Any], grad_clip_norm: Optional[float]):
    if "gradient_clipping" in config:
        ds_grad_clip_norm = config["gradient_clipping"]
        if ds_grad_clip_norm != grad_clip_norm:
            raise ValueError("Provided DeepSpeed configuration specifies grad clip norm="
                             f"{ds_grad_clip_norm}, but the Mosaic trainer has been configured "
                             f"with grad clip norm={grad_clip_norm}")

    if grad_clip_norm is not None:
        config["gradient_clipping"] = grad_clip_norm

    if "zero_allow_untested_optimizer" in config and not config["zero_allow_untested_optimizer"]:
        warnings.warn("Provided DeepSpeed configuration specifies zero_allow_untested_optimizer=False. "
                      "This causes DeepSpeed to reject certain Mosaic optimizers that are known to "
                      "work well with DeepSpeed.")

    config["zero_allow_untested_optimizer"] = True


def parse_deepspeed_config(config: Dict[str, Any],
                           state: State,
                           grad_clip_norm: Optional[float] = None) -> Dict[str, Any]:
    new_config = copy.deepcopy(config)
    parse_batch_settings(new_config, state)
    parse_precision_settings(new_config, state)
    parse_misc_settings(new_config, grad_clip_norm)
    return new_config


def _convert_fp32_tensor_to_fp16(tensor: Tensor):
    if tensor.dtype == torch.float32:
        return tensor.half()
    return tensor


def fix_batch_precision_for_deepspeed(batch: Batch, precision: Precision) -> Batch:
    """Ensures that a batch is properly formatted for DeepSpeed FP16, if active.

    This is more finnicky than it may sound. Just because we're in FP16 doesn't mean
    we can convert the entire batch to FP16 too. For example, integer tensors are common
    in inputs and outputs of various models, and these must not be converted. We make a
    big assumption that a tensor should only be converted to FP16 if it was given in FP32.
    """

    if precision != Precision.FP16:
        return batch

    return map_collection(batch, _convert_fp32_tensor_to_fp16)  # type: ignore


def is_module_deepspeed(module: torch.nn.Module) -> bool:
    """Returns whether the module is an instance of a deepspeed module.

    Args:
        module (torch.nn.Module): The module to check.

    Returns:
        bool: Whether the module is a deepspeed module.
    """
    try:
        import deepspeed
    except ImportError:
        return False
    else:
        return isinstance(module, deepspeed.DeepSpeedEngine)
