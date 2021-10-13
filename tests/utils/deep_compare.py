# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Any

import numpy as np
import torch


def deep_compare(a: Any, b: Any) -> None:
    if isinstance(a, (list, tuple)):
        assert isinstance(b, type(a))
        assert len(a) == len(b)
        for x, y in zip(a, b):
            deep_compare(x, y)
        return
    elif isinstance(a, dict):
        assert isinstance(b, dict)
        assert len(a) == len(b)
        for k, v in a.items():
            deep_compare(v, b[k])
        return
    elif isinstance(a, (torch.Tensor, np.ndarray)):
        assert isinstance(b, type(a))
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert (a == b).all()
        return
    else:
        assert a == b
