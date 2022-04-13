# Copyright 2021 MosaicML. All Rights Reserved.

"""Enum class for the numerical precision to be used by the model."""

from composer.utils.string_enum import StringEnum

__all__ = ["Precision"]


class Precision(StringEnum):
    """Enum class for the numerical precision to be used by the model.

    Attributes:
        AMP: Use :mod:`torch.cuda.amp`. Only compatible with GPUs.
        FP16: Use 16-bit floating-point precision. Currently only
            compatible with GPUs on DeepSpeed.
        FP32: Use 32-bit floating-point precision.
            Compatible with CPUs and GPUs.
        BF16: Use 16-bit BFloat mixed precision. Requires PyTorch
            1.10. Compatible with CPUs and GPUs.
    """
    AMP = "amp"
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
