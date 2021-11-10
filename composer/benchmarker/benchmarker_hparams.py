# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import yahp as hp

from composer.datasets.synthetic import SyntheticDataLabelType, validate_label_inputs
from composer.loggers import BaseLoggerBackendHparams
from composer.models import ModelHparams

if TYPE_CHECKING:
    from composer.benchmarker.benchmarker import Benchmarker


@dataclass
class BenchmarkerHparams(hp.Hparams):
    """Params for the :class:`~composer.benchmarker.benchmarker.Benchmarker`.

    See the documentation for the :class:`~composer.benchmarker.benchmarker.Benchmarker`.
    """

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
    data_shape: List[int] = hp.required("Shape of the data tensor.")
    loggers: List[BaseLoggerBackendHparams] = hp.required(doc="loggers to use")
    label_type: SyntheticDataLabelType = hp.optional("Type of synthetic label to create.",
                                                     default=SyntheticDataLabelType.CLASSIFICATION_INT)
    num_classes: int = hp.optional(
        "Number of classes. Required if label_type is SyntheticDataLabelType.CLASSIFICATION.", default=2)
    label_shape: List[int] = hp.optional(
        "Shape of the label tensor. Required if label_type is SyntheticDataLabelType.RANDOM_INT.",
        default_factory=lambda: [1])

    def validate(self):
        super().validate()

        validate_label_inputs(label_type=self.label_type, num_classes=self.num_classes, label_shape=self.label_shape)

    def initialize_object(self) -> Benchmarker:
        from composer.benchmarker.benchmarker import Benchmarker
        return Benchmarker.create_from_hparams(hparams=self)
