# Copyright 2021 MosaicML. All Rights Reserved.

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import yahp as hp

from composer.core import State
from composer.core.types import Precision


@dataclass
class DeepSpeedHparams(hp.Hparams):
    """Params for configuration of DeepSpeed."""

    enabled: bool = hp.required("Whether to use DeepSpeed.")

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
