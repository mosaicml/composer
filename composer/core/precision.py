# Copyright 2022 MosaicML. All Rights Reserved.

"""Enum class for the numerical precision to be used by the model."""

import contextlib
from typing import Generator, Union

import torch

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

    .. warning::

        :attr:`.Precision.FP16` is only supported when using DeepSpeed, as PyTorch does not
        natively support this precision. When this function is invoked with :attr:`.Precision.FP16`,
        the precision context will be a no-op.

    Args:
        precision (str | Precision): Precision for the context
    """

    precision = Precision(precision)
    if precision == Precision.FP32:
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', enabled=False):
                yield
        else:
            # Yield here to avoid warnings about cuda not being available
            yield
    elif precision == Precision.FP16:
        # No-op if FP16. FP16 is only supported by DeepSpeed, which is configured via the `deepspeed_config`
        # DeepSpeed ignores `get_precision_context`. The Trainer init validates that Precision.FP16 is used
        # only when using DeepSpeed.
        yield
    elif precision in (Precision.AMP, Precision.BF16):
        dtype = torch.bfloat16 if precision == Precision.BF16 else torch.float16
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
            yield
    else:
        raise ValueError(f"Unsupported precision: {precision}")
