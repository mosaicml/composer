# Copyright 2022 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams


@dataclass
class SSDHparams(ModelHparams):
    input_size: int = hp.optional(
        doc="input size",
        default=300,
    )
    num_classes: int = hp.optional(
        doc="num_classes",
        default=80,
    )

    overlap_threshold: float = hp.optional(
        doc="threshold",
        default=0.5,
    )

    nms_max_detections: int = hp.optional(
        doc="nms max dets",
        default=200,
    )
    data: str = hp.optional(
        doc="data",
        default="/localdisk/coco",
    )

    def initialize_object(self):
        from composer.models.ssd.ssd import SSD
        return SSD(
            input_size=self.input_size,
            overlap_threshold=self.overlap_threshold,
            nms_max_detections=self.nms_max_detections,
            num_classes=self.num_classes,
            data=self.data,
        )
