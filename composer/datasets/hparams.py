# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
import dataclasses
import textwrap
from typing import Callable, List, NamedTuple, Optional, Sequence, Type, Union, get_type_hints

import yahp as hp
from yahp.utils.type_helpers import HparamsType

from composer.core.types import Batch, DataLoader, TDeviceTransformFn, Tensor
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
class DatadirHparamsMixin(hp.Hparams, abc.ABC):
    """Datadir field mixing for :class:`DatasetHparams`.

    Parameters:
        datadir (str): The path to the data directory.
    """
    datadir: Optional[str] = hp.optional("The path to the data directory", default=None)


@dataclasses.dataclass
class SyntheticBatchesHparamsMixin(hp.Hparams, abc.ABC):
    """Synthic field mixing for :class:`DatasetHparams`.

    Parameters:
        synthetic (hp.Hparams, optional): Parameters to use for synthetic data generation. If None (the default),
            then real data will be used.
    """

    synthetic: Optional[hp.Hparams] = hp.optional(textwrap.dedent("""Parameters to use for synthetic data generation.
        If None (the default), then real data will be used."""),
                                                  default=None)

    @classmethod
    def get_synthetic_hparams_cls(cls) -> Type[hp.Hparams]:
        """get_synthetic_hparams_cls returns the type of the dataclass for the
        field :attr:`synthetic`.

        Raises:
            NotImplementedError: If the dataset hparams does not support synthetic data

        Returns:
            Type[hp.Hparams]: The type of the field :attr:`synthetic`.
        """
        field_types = get_type_hints(cls)
        for field in dataclasses.fields(cls):
            if field.name == "synthetic":
                hparams_type = HparamsType(field_types[field.name])
                return hparams_type.type
        raise RuntimeError(f"Invariant error -- Dataset {cls.__name__} does not support synthetic data")


@dataclasses.dataclass
class NumTotalBatchesHparamsMixin(hp.Hparams, abc.ABC):
    """Num total batches field mixing for :class:`DatasetHparams`.

    Parameters:
        num_total_batches (int, optional): If specified, then the dataloader from :meth:`initialize_object`
            should yield this many batches per iteration. Specifically, `len(dataloader) == num_total_batches`.
            Each epoch should yield the same subset of samples.
            
            If this value is greater than the total number of samples in the dataset, then a :class:`ValueError` 
            may be raised.

            If None (the default), then the entire dataset will be iterated over.
    """
    num_total_batches: Optional[int] = hp.optional(
        "If not None, limit len(dataloader) to this many batches. If None (the default), then the dataloader will iterate over the entire dataset.",
        default=None)


@dataclasses.dataclass
class ShuffleHparamsMixin(hp.Hparams, abc.ABC):
    """Shuffle field mixing for :class:`DatasetHparams`.

    Parameters:
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.
    """
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch. Defaults to True.", default=True)


@dataclasses.dataclass
class DropLastHparamsMixin(hp.Hparams, abc.ABC):
    """Drop last field mixing for :class:`DatasetHparams`.

    Parameters:
        drop_last (bool): If the number of samples is not divisible by the batch size, whether
            to drop the last batch (the default) or pad the last batch with zeros.
    """
    drop_last: bool = hp.optional(textwrap.dedent("""If the number of samples is not divisible by the batch size,
        whether to drop the last batch (the default) or pad the last batch with zeros."""),
                                  default=True)


@dataclasses.dataclass
class IsTrainHparamsMixin(hp.Hparams, abc.ABC):
    """Is train field mixing for :class:`DatasetHparams`.

    Attributes:
        is_train (bool): Whether to load the training data (the default) or validation data.
    """
    is_train: bool = hp.optional("Whether to load the training data (the default) or validation data.", default=True)


@dataclasses.dataclass
class DatasetHparams(hp.Hparams, abc.ABC):
    """Abstract base class for hyperparameters to initialize a dataset."""

    @abc.abstractmethod
    def initialize_object(self, batch_size: int,
                          dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataloaderSpec]:
        """Initializes a :class:`DataloaderSpec` for this dataset.
        
        Parameters:
            batch_size (int): The size of the batch the dataloader should yield. This batch size is
                device-specific and already incorporates the world size.
            dataloader_hparams (DataloaderHparams): The dataset-independent hparams for the dataloader
        
        Returns:
            :class:`Dataloader` or :class:`DataloaderSpec`: The dataloader, or if a custom device transformation
            or split function is required, a :class:`DataloaderSpec` tuple
        """
        pass
