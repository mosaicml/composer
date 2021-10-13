# Copyright 2021 MosaicML. All Rights Reserved.

from composer.utils.string_enum import StringEnum


class Precision(StringEnum):
    """Enum class for the numerical precision to be used by the model."""

    AMP = "amp"
    FP32 = "fp32"
