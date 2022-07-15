# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous Helpers."""

import torch

__all__ = ['is_model_deepspeed']


def is_model_deepspeed(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`~deepspeed.DeepSpeedEngine`."""
    try:
        import deepspeed
    except ImportError:
        return False
    else:
        return isinstance(model, deepspeed.DeepSpeedEngine)
