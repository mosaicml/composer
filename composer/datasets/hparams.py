# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Optional, Sequence, Set, Type, Union, get_type_hints

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
    
    Attributes:
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
class DatasetHparams(hp.Hparams, ABC):
    """Abstract base class for hyperparameters to initialize a dataset.
    
    If the dataset supports generating synthetic data, add a "synthetic" field to the hparams.
    If this field is True, then the dataloader should yield samples from a synthetic (randomly generated)
    dataset that does not depend on the real dataset.
    """

    # declaring default values without type annotations for these common fields
    # This ensure that they are not registered as fields in the dataclass, but they register in Pyright
    synthetic = None  # type: Optional[hp.Hparams]
    num_total_batches = None  # type: Optional[int]
    shuffle = False  # type: bool
    drop_last = False  # type: bool
    datadir = ""  # type: str

    def __post_init__(self):
        # Ensure that NotImplementedErrors are raised when attempting to use these common fields on unsupported datasets
        fields = dataclasses.fields(self)
        common_fields = ["synthetic", "num_total_batches", "shuffle", "drop_last", "datadir"]
        field_names: Set[str] = set(f.name for f in fields)
        for common_field in common_fields:
            if common_field not in field_names:
                def handler(field_name=common_field, *args, **kwargs):
                    del args, kwargs  # unused
                    raise AttributeError(f"Dataset {self.__class__.__name__} does not support {field_name}")
                setattr(self, common_field, property(fget=handler, fset=handler, fdel=lambda x: None, doc=common_field))

    @classmethod
    def get_synthetic_hparams_cls(cls) -> Type[hp.Hparams]:
        field_types = get_type_hints(cls)
        for field in dataclasses.fields(cls):
            if field.name == "synthetic":
                hparams_type = HparamsType(field_types[field.name])
                if not hparams_type.is_hparams_dataclass:
                    raise NotImplementedError(f"Dataset {cls.__name__} does not support synthetic data")
                return hparams_type.type
        raise NotImplementedError(f"Dataset {cls.__name__} does not support synthetic data")

    @abstractmethod
    def initialize_object(self, batch_size: int,
                          dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataloaderSpec]:
        """Initializes a :class:`DataloaderSpec` for this dataset.
        
        Args:
            batch_size (int): The size of the batch the dataloader should yield
            dataloader_hparams (DataloaderHparams): The dataset-independent hparams for the dataloader
        
        Returns:
            :class:`Dataloader` or :class:`DataloaderSpec`: The dataloader, or if a custom device transformation
            or split function is required, a :class:`DataloaderSpec` tuple
        """
        pass
