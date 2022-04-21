# Copyright 2021 MosaicML. All Rights Reserved.

"""Enum class for the numerical precision to be used by the model."""

import contextlib
from typing import Generator, Union

import torch
from packaging import version

from composer.utils.string_enum import StringEnum

__all__ = ["Precision", "get_precision_context"]


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


@contextlib.contextmanager
def get_precision_context(precision: Union[str, Precision]) -> Generator[None, None, None]:
    """Returns a context manager to automatically cast to a specific precision.

    Args:
        precision (str or Precision): Precision for the context
    """

    precision = Precision(precision)
    enabled = False
    if precision == Precision.FP32:
        if not torch.cuda.is_available():
            # Yield here to avoid warnings about cuda not being available
            yield
            return
        enabled = False
    elif precision == Precision.AMP:
        enabled = True
    elif precision == Precision.BF16:
        if version.parse(torch.__version__) < version.parse("1.10"):
            raise ValueError(f"BF16 precision requires torch > 1.10, got version {torch.__version__}")
        with torch.cuda.amp.autocast(True, torch.bfloat16):  # type: ignore
            yield
        # Retain compatibility with PyTorch < 1.10
        if precision != Precision.BF16:
            with torch.cuda.amp.autocast(enabled):  # type: ignore
                yield
