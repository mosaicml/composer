# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Enum class for the numerical precision to be used by the model."""

import contextlib
import os
from typing import Generator, Union

import torch

from composer.utils import StringEnum

__all__ = ['Precision', 'get_precision_context']


class Precision(StringEnum):
    """Enum class for the numerical precision to be used by the model.

    Attributes:
        FP32: Use 32-bit floating-point precision. Compatible with CPUs and GPUs.
        AMP_FP16: Use :mod:`torch.cuda.amp` wih 16-bit floating-point precision. Only compatible
            with GPUs.
        AMP_BF16: Use :mod:`torch.cuda.amp` wih 16-bit BFloat precision.
    """
    FP32 = 'fp32'
    AMP_FP16 = 'amp_fp16'
    AMP_BF16 = 'amp_bf16'


@contextlib.contextmanager
def get_precision_context(precision: Union[str, Precision]) -> Generator[None, None, None]:
    """Returns a context manager to automatically cast to a specific precision.

    Args:
        precision (str | Precision): Precision for the context
    """
    precision = Precision(precision)
    if precision == Precision.FP32:
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(False):
                yield
        else:
            # Yield here to avoid warnings about cuda not being available
            yield
    elif precision == Precision.AMP_FP16:
        # Retain compatibility with PyTorch < 1.10
        with torch.cuda.amp.autocast(True):
            yield
    elif precision == Precision.AMP_BF16:
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                yield
        else:
            os.environ['XLA_USE_BF16'] = '1'
            yield
    else:
        raise ValueError(f'Unsupported precision: {precision}')
