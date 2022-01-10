# Copyright 2021 MosaicML. All Rights Reserved.

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import yahp as hp

from composer.core import State
from composer.core.types import Batch, Precision, Tensor
from composer.utils.iter_helpers import map_collection


@dataclass
class DeepSpeedHparams(hp.Hparams):
    """Params for configuration of DeepSpeed."""

    zero_stage: int = hp.optional("The ZeRO memory optimization stage to use.", default=0)

    optimizer_offload: bool = hp.optional(
        "Whether to offload the optimizer to CPU. Compatible only with zero_stage >= 2.", default=False)

    parameter_offload: bool = hp.optional(
        "Whether to offload model parameters to CPU. Compatible only with zero_stage = 3 and optimizer_offload.",
        default=False)

    zero2_bucket_size: int = hp.optional("Buffer size used by ZeRO 2 for distributed communications.", default=int(5e8))
    overlap_comm: bool = hp.optional("Overlap comm", default=False)

    gradient_checkpointing: bool = hp.optional("Whether to enable gradient checkpointing.", default=False)

    def validate(self):
        super().validate()

        if self.zero_stage not in [0, 1, 2, 3]:
            raise ValueError("DeepSpeed ZeRO stage must be one of [0, 1, 2, 3].")

        if self.optimizer_offload and self.zero_stage < 2:
            raise ValueError("DeepSpeed optimizer offloading is only compatible with ZeRO stage >= 2.")

        if self.parameter_offload and self.zero_stage != 3:
            raise ValueError("DeepSpeed parameter offloading is only compatible with ZeRO stage = 3.")

        if self.parameter_offload and not self.optimizer_offload:
            raise ValueError(
                "DeepSpeed parameter offloading is only supported when optimizer offloading is also enabled.")

        if self.zero_stage == 3:
            warnings.warn("ZeRO stage 3 is largely untested with composer. Certain algorithms may break.")

    def initialize_object(self, state: State, grad_clip_norm: Optional[float]):

        deepspeed_config: dict[str, Any] = {
            "train_batch_size": state.train_batch_size,
            "gradient_accumulation_steps": state.grad_accum,
            "zero_optimization": {
                "stage": self.zero_stage,
                "allgather_bucket_size": self.zero2_bucket_size,
                "reduce_bucket_size": self.zero2_bucket_size,
                "overlap_comm": self.overlap_comm,
            },

            # Without this, DeepSpeed throws errors when ZeRO is used in combination with
            # non-standard optimizers. Most likely, this will trigger when one of the decoupled
            # weight decay optimizers is used, but it has been verified that those optimizers work
            # in combination with DeepSpeed.
            "zero_allow_untested_optimizer": True,
        }

        if self.optimizer_offload:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
            }

        if self.parameter_offload:
            deepspeed_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
            }

        if state.precision == Precision.AMP:
            deepspeed_config["amp"] = {"enabled": True}
        elif state.precision == Precision.FP16:
            deepspeed_config["fp16"] = {
                "enabled": True,
                "initial_scale_power": 16,
                "loss_scale_window": 2000,
            }

        if grad_clip_norm:
            deepspeed_config["gradient_clipping"] = grad_clip_norm

        return deepspeed_config


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
