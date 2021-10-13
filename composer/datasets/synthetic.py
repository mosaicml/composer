# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.utils.data
import yahp as hp

from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.utils.string_enum import StringEnum


class SyntheticDataType(StringEnum):
    INCREASING = "increasing"
    GAUSSIAN = "gaussian"
    SEPARABLE = "separable"


class MemoryFormat(StringEnum):
    CONTIGUOUS_FORMAT = "contiguous_format"
    CHANNELS_LAST = "channels_last"
    CHANNELS_LAST_3D = "channels_last_3d"
    PRESERVE_FORMAT = "preserve_format"


class SyntheticDataset(torch.utils.data.Dataset):

    def __init__(self,
                 *,
                 sample_pool_size: int,
                 shape: Sequence[int],
                 memory_format: Union[str, MemoryFormat],
                 device: str,
                 one_hot: bool,
                 num_classes: int,
                 transform: Optional[Callable] = None,
                 data_type: SyntheticDataType = SyntheticDataType.GAUSSIAN):
        self.size = sample_pool_size
        self.shape = shape
        self.input_data = None
        self.input_target = None
        self.memory_format = getattr(torch, MemoryFormat(memory_format).value)
        self.device = device
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.transform = transform
        self.data_type = data_type

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if self.input_data is None:
            # Generating data on the first call to __getitem__ so that data is stored on the correct gpu,
            # after DeviceSingleGPU calls torch.cuda.set_device
            # This does mean that the first batch will be slower
            # generating samples so all values for the sample are the sample index
            # e.g. all(input_data[1] == 1). Helps with debugging.
            assert self.input_target is None
            if self.data_type == SyntheticDataType.GAUSSIAN or \
                self.data_type == SyntheticDataType.SEPARABLE:
                input_data = torch.randn(self.size, *self.shape)
            else:
                input_data = torch.arange(start=0, end=self.size, step=1, dtype=torch.float)
                input_data = input_data.reshape(self.size, *(1 for _ in self.shape))
                input_data = input_data.expand(self.size, *self.shape)  # returns a view
            assert input_data.shape == (self.size, *self.shape)
            input_data = torch.clone(input_data)  # allocate actual memory
            input_data = input_data.contiguous(memory_format=self.memory_format)
            input_data = input_data.to(self.device)
            if self.one_hot:
                input_target = torch.empty(self.size, self.num_classes).to(self.device)
                input_target[:, 0] = 1.0
            else:
                input_target = torch.randint(0, self.num_classes, (self.size,))

            # If separable, force the positive examples to have a higher mean than the negative examples
            if self.data_type == SyntheticDataType.SEPARABLE:
                assert not self.one_hot, "SyntheticDataType.SEPARABLE does not support one_hot=True."
                assert max(input_target) == 1 and min(input_target) == 0, \
                    "SyntheticDataType.SEPARABLE only supports binary labels"
                # Make positive examples have mean = 3 and negative examples have mean = -3
                # so they are easier to separate with a classifier
                input_data[input_target == 0] -= 3
                input_data[input_target == 1] += 3

            self.input_data = input_data
            self.input_target = input_target
        assert self.input_target is not None
        if self.transform is not None:
            return self.transform(self.input_data[idx]), self.input_target[idx]
        else:
            return self.input_data[idx], self.input_target[idx]


@dataclass
class SyntheticDatasetHparams(DatasetHparams):
    """Defines an instance of a synthetic dataset for classification.
    
    Parameters:
        num_classes (int): Number of classes to use.
        shape (List[int]): Shape of the tensor for input samples.
        one_hot (bool): Whether to use one-hot encoding.
        device (str): Device to store the sample pool. Set to `cuda` to store samples
            on the GPU and eliminate PCI-e bandwidth with the dataloader. Set to `cpu`
            to move data between host memory and the gpu on every batch.
        memory_format (MemoryFormat, optional): Memory format for the sample pool.
        sample_pool_size (int): Number of samples to use.
        drop_last (bool): Whether to drop the last samples for the last batch.
        shuffle (bool): Whether to shuffle the dataset for each epoch.
        data_type (SyntheticDataType, optional), Type of synthetic data to create.
    """

    num_classes: int = hp.required("Number of classes")
    shape: List[int] = hp.required("Shape of tensor")
    one_hot: bool = hp.required("Whether to use one-hot encoding", template_default=False)
    device: str = hp.required(
        "Device to store the sample pool. "
        "Set to `cuda` to store samples on the GPU and eliminate PCI-e bandwidth with the dataloader. "
        "Set to `cpu` to move data between host memory and the gpu on every batch. ",
        template_default="cpu")
    memory_format: MemoryFormat = hp.optional("Memory format for the sample pool",
                                              default=MemoryFormat.CONTIGUOUS_FORMAT)
    sample_pool_size: int = hp.optional("Number of samples", default=100)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)
    data_type: SyntheticDataType = hp.optional("Type of synthetic data to create.", default=SyntheticDataType.GAUSSIAN)

    def initialize_object(self) -> DataloaderSpec:
        return DataloaderSpec(
            SyntheticDataset(
                num_classes=self.num_classes,
                shape=self.shape,
                one_hot=self.one_hot,
                device=self.device,
                memory_format=self.memory_format,
                sample_pool_size=self.sample_pool_size,
                data_type=self.data_type,
            ),
            drop_last=self.drop_last,
            shuffle=False,
        )
