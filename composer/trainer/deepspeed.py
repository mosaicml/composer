from dataclasses import dataclass

import yahp as hp


@dataclass
class DeepSpeedHparams(hp.Hparams):
    """Params for configuration of DeepSpeed."""

    enabled: bool = hp.required("Whether to use DeepSpeed.")

    def validate(self):
        super().validate()
