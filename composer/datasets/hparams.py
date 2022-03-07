# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
import dataclasses
import textwrap
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
from composer.datasets.dataloader import DataloaderHparams

__all__ = ["SyntheticHparamsMixin", "DatasetHparams"]


@dataclasses.dataclass
class SyntheticHparamsMixin(hp.Hparams, abc.ABC):
    """Synthetic dataset parameter mixin for :class:`DatasetHparams`.

    Parameters:
        use_synthetic (bool, optional): Whether to use synthetic data. (Default: ``False``)
        synthetic_num_unique_samples (int, optional): The number of unique samples to allocate memory for.
            Ignored if :attr:`use_synthetic` is False. (Default: ``100``)
        synthetic_device (str, optonal): The device to store the sample pool.
            Set to ``cuda`` to store samples on the GPU and eliminate PCI-e bandwidth with the dataloader.
            Set to ``cpu`` to move data between host memory and the device on every batch.
            Ignored if :attr:`use_synthetic` is False. (Default: ``cpu``)
        synthetic_memory_format: The :class:`MemoryFormat` to use.
            Ignored if :attr:`use_synthetic` is False. (Default: ``CONTIGUOUS_FORMAT``)
    """

    use_synthetic: bool = hp.optional("Whether to use synthetic data. Defaults to False.", default=False)
    synthetic_num_unique_samples: int = hp.optional("The number of unique samples to allocate memory for.", default=100)
    synthetic_device: str = hp.optional("Device to store the sample pool. Should be `cuda` or `cpu`. Defauls to `cpu`.",
                                        default="cpu")
    synthetic_memory_format: MemoryFormat = hp.optional("Memory format. Defaults to contiguous format.",
                                                        default=MemoryFormat.CONTIGUOUS_FORMAT)


@dataclasses.dataclass
class DatasetHparams(hp.Hparams, abc.ABC, metaclass=metaclass):
    """Abstract base class for hyperparameters to initialize a dataset.

    Parameters:
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data (the default) or validation data.
        drop_last (bool):
            If the number of samples is not divisible by the batch size, whether
            to drop the last batch (the default) or pad the last batch with zeros.
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.
    """

    is_train: bool = hp.optional("Whether to load the training data (the default) or validation data.", default=True)
    drop_last: bool = hp.optional(textwrap.dedent("""\
        If the number of samples is not divisible by the batch size,
        whether to drop the last batch (the default) or pad the last batch with zeros."""),
                                  default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch. Defaults to True.", default=True)

    datadir: Optional[str] = hp.optional("The path to the data directory", default=None)

    @abc.abstractmethod
    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataSpec]:
        """Creates a :class:`DataLoader` or :class:`DataloaderSpec` for this dataset.

        Parameters:
            batch_size (int): The size of the batch the dataloader should yield. This batch size is
                device-specific and already incorporates the world size.
            dataloader_hparams (DataloaderHparams): The dataset-independent hparams for the dataloader

        Returns:
            Dataloader or DataSpec: The dataloader, or if the dataloader yields batches of custom types,
            a :class:`DataSpec`.
        """
        pass
