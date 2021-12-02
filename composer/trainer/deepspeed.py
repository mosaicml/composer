# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp


@dataclass
class DeepSpeedHparams(hp.Hparams):
    """Params for configuration of DeepSpeed."""

    enabled: bool = hp.required("Whether to use DeepSpeed.")

    zero_stage: int = hp.optional("The ZeRO memory optimization stage to use.", default=0)

    def validate(self):
        super().validate()

        if self.zero_stage not in [0, 1, 2, 3]:
            raise ValueError("DeepSpeed ZeRO stage must be one of [0, 1, 2, 3].")
