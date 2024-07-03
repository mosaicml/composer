# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Reference for common types used throughout the composer library.

Attributes:
    Batch (Any): Alias to type Any.
        A batch of data can be represented in several formats, depending on the application.
    JSON (str | float | int | None | list['JSON'] | dict[str, 'JSON']): JSON Data.
    Dataset (torch.utils.data.Dataset[Batch]): Alias for :class:`torch.utils.data.Dataset`.
"""

from __future__ import annotations

from typing import Any, Union

import torch
import torch.utils.data

from composer.utils import StringEnum

__all__ = ['Batch', 'JSON', 'MemoryFormat', 'TrainerMode']

Batch = Any

Dataset = torch.utils.data.Dataset[Batch]

JSON = Union[str, float, int, None, list['JSON'], dict[str, 'JSON']]


class TrainerMode(StringEnum):
    """Enum to represent which mode the Trainer is in.

    Attributes:
        TRAIN: In training mode.
        EVAL: In evaluation mode.
        PREDICT: In predict mode.
    """
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'


class MemoryFormat(StringEnum):
    """Enum class to represent different memory formats.

    See :class:`torch.torch.memory_format` for more details.

    Attributes:
        CONTIGUOUS_FORMAT: Default PyTorch memory format represnting a tensor allocated with consecutive dimensions
            sequential in allocated memory.
        CHANNELS_LAST: This is also known as NHWC. Typically used for images with 2 spatial dimensions (i.e., Height and
            Width) where channels next to each other in indexing are next to each other in allocated memory. For example, if
            C[0] is at memory location M_0 then C[1] is at memory location M_1, etc.
        CHANNELS_LAST_3D: This can also be referred to as NTHWC. Same as :attr:`CHANNELS_LAST` but for videos with 3
            spatial dimensions (i.e., Time, Height and Width).
        PRESERVE_FORMAT: A way to tell operations to make the output tensor to have the same memory format as the input
            tensor.
    """
    CONTIGUOUS_FORMAT = 'contiguous_format'
    CHANNELS_LAST = 'channels_last'
    CHANNELS_LAST_3D = 'channels_last_3d'
    PRESERVE_FORMAT = 'preserve_format'
