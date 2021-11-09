# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

from dataclasses import dataclass

import yahp as hp

from composer.models import ModelHparams


@dataclass
class BenchmarkerHparams(hp.Hparams):

    model: ModelHparams = hp.required(doc="model")
    total_batch_size: int = hp.required(
        doc="batch size for each optimization step, across all devices and gradient accumulations.",
        template_default=2048,
    )

    grad_accum: int = hp.required(
        template_default=1,
        doc=
        "Determines the number of microbatches to split a per-gpu batch into, used to compensate for low-memory-capacity devices."
    )
