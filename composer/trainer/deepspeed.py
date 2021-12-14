# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp


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
