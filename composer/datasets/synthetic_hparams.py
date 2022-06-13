# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Dataset hyperparameter mixin."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import yahp as hp

from composer.core.types import MemoryFormat

__all__ = ['SyntheticHparamsMixin']


@dataclass
class SyntheticHparamsMixin(hp.Hparams, abc.ABC):
    """Synthetic dataset parameter mixin for :class:`DatasetHparams`.

    If a dataset supports yielding synthetic data, it should implement this mixin.

    Args:
        use_synthetic (bool, optional): Whether to use synthetic data. Default: ``False``.
        synthetic_num_unique_samples (int, optional): The number of unique samples to
            allocate memory for. Ignored if :attr:`use_synthetic` is ``False``. Default:
            ``100``.
        synthetic_device (str, optional): The device to store the sample pool on.
            Set to ``'cuda'`` to store samples on the GPU and eliminate PCI-e bandwidth
            with the dataloader. Set to ``'cpu'`` to move data between host memory and the
            device on every batch. Ignored if :attr:`use_synthetic` is ``False``. Default:
            ``'cpu'``.
        synthetic_memory_format: The :class:`~.core.types.MemoryFormat` to use.
            Ignored if :attr:`use_synthetic` is ``False``. Default:
            ``'CONTIGUOUS_FORMAT'``.
    """

    use_synthetic: bool = hp.optional('Whether to use synthetic data. Defaults to False.', default=False)
    synthetic_num_unique_samples: int = hp.optional('The number of unique samples to allocate memory for.', default=100)
    synthetic_device: str = hp.optional('Device to store the sample pool. Should be `cuda` or `cpu`. Defauls to `cpu`.',
                                        default='cpu')
    synthetic_memory_format: MemoryFormat = hp.optional('Memory format. Defaults to contiguous format.',
                                                        default=MemoryFormat.CONTIGUOUS_FORMAT)
