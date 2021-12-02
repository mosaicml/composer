# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
import dataclasses
import textwrap
from typing import Callable, List, NamedTuple, Optional, Sequence, Union

try:
    import custom_inherit
except ImportError:
    # if custom_inherit is not installed, then the docstrings will be incomplete. That's fine.
    metaclass = abc.ABCMeta
else:
    metaclass = custom_inherit.DocInheritMeta(style="google_with_merge", abstract_base_class=True)

import yahp as hp

from composer.core.types import Batch, DataLoader, MemoryFormat, TDeviceTransformFn, Tensor
from composer.datasets.dataloader import DataloaderHparams


def _split_fn(batch: Batch, n_microbatches: int) -> List[Batch]:
    if not isinstance(batch, Sequence):
        raise ValueError(f'split_fn requires batch be a tuple pair of tensors, got {type(batch)}')
    x, y = batch
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return list(zip(x.chunk(n_microbatches), y.chunk(n_microbatches)))
    if isinstance(x, List) and isinstance(y, List):
        return list(
            zip(
                [x[i::n_microbatches] for i in range(n_microbatches)],
                [y[i::n_microbatches] for i in range(n_microbatches)],
            ))
    raise NotImplementedError('The default split_fn is unable to split the output of this'
                              'dataloader. Please define a split_fn in your dataloader spec.')


class DataloaderSpec(NamedTuple):
    """Specification for initializing a dataloader when a device transformation function or split function
    is required
    
    Parameters:
        dataloader (DataLoader): The initialized dataloader.
        device_transform_fn (TDeviceTransformFn, optional):
            A function to modify the data once it has been loaded onto the device (for example, GPU-based batch normalization)
            This function is invoked with a batch of data after it has been moved onto the device,
            and it is expected to return a batch.
        split_fn (Batch, int -> List[Batch]): A function to
            run to split batches into microbatches.
    """
    dataloader: DataLoader
    device_transform_fn: Optional[TDeviceTransformFn] = None
    split_fn: Callable[[Batch, int], List[Batch]] = _split_fn


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

    use_synthetic: bool = hp.optional("Whether to use synthetic data. Defaults to False." "", default=False)
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
        subset_num_batches (int, optional): If specified, limit the number of batches per dataloader iteration.
            Specifically, ``len(dataloader) == num_total_batches``, where the ``dataloader`` is returned via
            :meth:`initialize_object`. Each epoch should yield the same subset of samples.
            
            If this value is greater than the total number of samples in the dataset, then a :class:`ValueError` 
            is raised.

            If None (the default), then the entire dataset will be iterated over.
    """

    is_train: bool = hp.optional("Whether to load the training data (the default) or validation data.", default=True)
    drop_last: bool = hp.optional(textwrap.dedent("""If the number of samples is not divisible by the batch size,
        whether to drop the last batch (the default) or pad the last batch with zeros."""),
                                  default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch. Defaults to True.", default=True)

    subset_num_batches: Optional[int] = hp.optional(
        "If not None, limit len(dataloader) to this many batches. If None (the default), then the dataloader will iterate over the entire dataset.",
        default=None)
    datadir: Optional[str] = hp.optional("The path to the data directory", default=None)

    @abc.abstractmethod
    def initialize_object(self, batch_size: int,
                          dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataloaderSpec]:
        """Creates a :class:`DataLoader` or :class:`DataloaderSpec` for this dataset.
        
        Parameters:
            batch_size (int): The size of the batch the dataloader should yield. This batch size is
                device-specific and already incorporates the world size.
            dataloader_hparams (DataloaderHparams): The dataset-independent hparams for the dataloader
        
        Returns:
            Dataloader or DataloaderSpec: The dataloader, or if a custom device transformation
                or split function is required, a :class:`DataloaderSpec` tuple
        """
        pass
