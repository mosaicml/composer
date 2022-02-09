# Copyright 2021 MosaicML. All Rights Reserved.

import contextlib
from typing import Callable, ContextManager, Union
from packaging import version

import torch
import torch.cuda.amp

from composer.core.precision import Precision


def default_precision_factory() -> Callable[[Union[str, Precision]], ContextManager]:
    """Returns a context manager to automatically cast to a specific precision

    Args:
        precision (str or Precision): Precision for the context
    """
    
    if torch.cuda.is_available():

        def cuda_precision(precision):
            if Precision(precision) == Precision.BF16:
                assert version.parse(torch.__version__) >= version.parse("1.10"), "Bfloat16 is only available for PyTorch versions >= 1.10" 
                return torch.cuda.amp.autocast(True, dtype=torch.bfloat16)
            return lambda precision: torch.cuda.amp.autocast(Precision(precision) == Precision.AMP)
        
        return cuda_precision
    else:

        def cpu_precision(precision):
            assert Precision(
                precision) != Precision.AMP, "Precision AMP is only available when `torch.cuda.is_available() == True`."
            if Precision(precision) == Precision.BF16:
                assert version.parse(torch.__version__) >= version.parse("1.10"), "Bfloat16 is only available for PyTorch versions >= 1.10"
                return torch.autocast(device_type='cpu', enabled=True, dtype=torch.bfloat16)
            return contextlib.nullcontext()

        return cpu_precision
