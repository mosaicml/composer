# Copyright 2021 MosaicML. All Rights Reserved.

"""Dataset Hyperparameter classes."""

from __future__ import annotations

import abc
import textwrap
from dataclasses import dataclass
from typing import Optional, Union

try:
    import custom_inherit
except ImportError:
    # if custom_inherit is not installed, then the docstrings will be incomplete. That's fine.
    metaclass = abc.ABCMeta
else:
    metaclass = custom_inherit.DocInheritMeta(style="google_with_merge", abstract_base_class=True)

import yahp as hp

from composer.core.types import DataLoader, DataSpec, MemoryFormat
from composer.datasets.dataloader import DataLoaderHparams

__all__ = ["SyntheticHparamsMixin", "DatasetHparams"]


@dataclass
class SyntheticHparamsMixin(hp.Hparams, abc.ABC):
    """Synthetic dataset parameter mixin for :class:`DatasetHparams`.

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

    use_synthetic: bool = hp.optional("Whether to use synthetic data. Defaults to False.", default=False)
    synthetic_num_unique_samples: int = hp.optional("The number of unique samples to allocate memory for.", default=100)
    synthetic_device: str = hp.optional("Device to store the sample pool. Should be `cuda` or `cpu`. Defauls to `cpu`.",
                                        default="cpu")
    synthetic_memory_format: MemoryFormat = hp.optional("Memory format. Defaults to contiguous format.",
                                                        default=MemoryFormat.CONTIGUOUS_FORMAT)


@dataclass
class DatasetHparams(hp.Hparams, abc.ABC, metaclass=metaclass):
    """Abstract base class for hyperparameters to initialize a dataset.

    Args:
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool): If the number of samples is not divisible by the batch size,
            whether to drop the last batch or pad the last batch with zeros. Default:
            ``True``.
        shuffle (bool): Whether to shuffle the dataset. Default: ``True``.
    """

    is_train: bool = hp.optional("Whether to load the training data (the default) or validation data.", default=True)
    drop_last: bool = hp.optional(textwrap.dedent("""\
        If the number of samples is not divisible by the batch size,
        whether to drop the last batch (the default) or pad the last batch with zeros."""),
                                  default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch. Defaults to True.", default=True)

    datadir: Optional[str] = hp.optional("The path to the data directory", default=None)

    @abc.abstractmethod
    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> Union[DataLoader, DataSpec]:
        """Creates a :class:`~.core.types.DataLoader` or
        :class:`~.core.data_spec.DataSpec` for this dataset.

        Args:
            batch_size (int): The size of the batch the dataloader should yield. This
                batch size is device-specific and already incorporates the world size.
            dataloader_hparams (DataLoaderHparams): The dataset-independent hparams for
                the dataloader.

        Returns:
            DataLoader or DataSpec: The :class:`~core.types.DataLoader`, or if the dataloader yields batches of custom
                types, a :class:`~core.data_spec.DataSpec`.
        """
        pass


@dataclass
class WebDatasetHparams(DatasetHparams, abc.ABC, metaclass=metaclass):
    """Abstract base class for hyperparameters to initialize a webdataset.

    Args:
        webdataset_cache_dir (str): WebDataset cache directory.
        webdataset_cache_verbose (str): WebDataset cache verbosity.
    """

    webdataset_cache_dir: str = hp.optional('WebDataset cache directory', default='/tmp/webdataset_cache/')
    webdataset_cache_verbose: bool = hp.optional('WebDataset cache verbosity', default=False)
    shuffle_buffer: int = hp.optional('WebDataset shuffle buffer size per worker', default=256)
