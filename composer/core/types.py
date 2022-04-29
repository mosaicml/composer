# Copyright 2021 MosaicML. All Rights Reserved.

"""Reference for common types used throughout the composer library.

Attributes:
    Batch (BatchPair | BatchDict | torch.Tensor): Union type covering the most common representations of batches.
        A batch of data can be represented in several formats, depending on the application.
    BatchPair (Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]]): Commonly used in computer vision tasks.
        The object is assumed to contain exactly two elements, where the first represents inputs
        and the second represents targets.
    BatchDict (Dict[str, Tensor]): Commonly used in natural language processing tasks.
    PyTorchScheduler (torch.optim.lr_scheduler._LRScheduler): Alias for base class of learning rate schedulers such
        as :class:`torch.optim.lr_scheduler.ConstantLR`.
    JSON (str | float | int | None | List['JSON'] | Dict[str, 'JSON']): JSON Data.
    Dataset (torch.utils.data.Dataset[Batch]): Alias for :class:`torch.utils.data.Dataset`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union
from typing import Dict, List, Union

import torch
import torch.utils.data

from composer.utils.string_enum import StringEnum

__all__ = ["Batch", "PyTorchScheduler", "JSON", "MemoryFormat", "BreakEpochException"]

Batch = Any

Dataset = torch.utils.data.Dataset[Batch]

PyTorchScheduler = torch.optim.lr_scheduler._LRScheduler

JSON = Union[str, float, int, None, List['JSON'], Dict[str, 'JSON']]


class BreakEpochException(Exception):
    """Raising this exception will immediately end the current epoch.

    If you're wondering whether you should use this, the answer is no.
    """

    pass


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
    CONTIGUOUS_FORMAT = "contiguous_format"
    CHANNELS_LAST = "channels_last"
    CHANNELS_LAST_3D = "channels_last_3d"
    PRESERVE_FORMAT = "preserve_format"
