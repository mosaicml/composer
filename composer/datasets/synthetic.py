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


class SyntheticDataLabelType(StringEnum):
    CLASSIFICATION = "classification"
    RANDOM_INT = "random_int"


class MemoryFormat(StringEnum):
    CONTIGUOUS_FORMAT = "contiguous_format"
    CHANNELS_LAST = "channels_last"
    CHANNELS_LAST_3D = "channels_last_3d"
    PRESERVE_FORMAT = "preserve_format"


'''
    sample_pool_size -> batch size
    batches_in_dataset (default 1)

    X -> leave as-is

    Y -> y_shape (default (1,))
    y_type: Classification, Random
    if classification - one_hot + num_classes
    if random - shape
'''


class SyntheticDataset(torch.utils.data.Dataset):

    def __init__(self,
                 *,
                 batch_size: int,
                 data_shape: Sequence[int],
                 batches_in_dataset: int = 1,
                 data_type: SyntheticDataType = SyntheticDataType.GAUSSIAN,
                 label_type: SyntheticDataLabelType = SyntheticDataLabelType.CLASSIFICATION,
                 one_hot: Optional[bool] = None,
                 num_classes: Optional[int] = None,
                 label_shape: Optional[Sequence[int]] = None,
                 device: str = "cpu",
                 memory_format: Union[str, MemoryFormat] = MemoryFormat.CONTIGUOUS_FORMAT,
                 transform: Optional[Callable] = None):
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.batches_in_dataset = batches_in_dataset
        self.data_type = data_type
        self.label_type = label_type
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.label_shape = label_shape
        self.device = device
        self.memory_format = getattr(torch, MemoryFormat(memory_format).value)
        self.transform = transform

        _validate_label_inputs(label_type=self.label_type,
                               num_classes=self.num_classes,
                               one_hot=self.one_hot,
                               label_shape=self.label_shape)

        # The synthetic data
        self.input_data = None
        self.input_target = None

    def __len__(self) -> int:
        return self.batch_size * self.batches_in_dataset

    def __getitem__(self, idx: int):
        idx = idx % self.batch_size
        if idx == 0 and self.input_data is not None and self.input_target is not None:
            # allocate actual memory on each new batch for more realistic
            # throughput
            self.input_data = torch.clone(self.input_data)
            self.input_target = torch.clone(self.input_target)

        if self.input_data is None:
            # Generating data on the first call to __getitem__ so that data is stored on the correct gpu,
            # after DeviceSingleGPU calls torch.cuda.set_device
            # This does mean that the first batch will be slower
            # generating samples so all values for the sample are the sample index
            # e.g. all(input_data[1] == 1). Helps with debugging.
            assert self.input_target is None
            if self.data_type == SyntheticDataType.GAUSSIAN or \
                self.data_type == SyntheticDataType.SEPARABLE:
                input_data = torch.randn(self.batch_size, *self.data_shape, device=self.device)
            elif self.data_type == SyntheticDataType.INCREASING:
                input_data = torch.arange(start=0, end=self.batch_size, step=1, dtype=torch.float, device=self.device)
                input_data = input_data.reshape(self.batch_size, *(1 for _ in self.data_shape))
                input_data = input_data.expand(self.batch_size, *self.data_shape)  # returns a view
            else:
                raise ValueError(f"Unsupported data type {self.data_type}")

            input_data = input_data.contiguous(memory_format=self.memory_format)

            if self.label_type == SyntheticDataLabelType.CLASSIFICATION:
                if self.one_hot:
                    input_target = torch.empty(self.batch_size, self.num_classes, device=self.device)
                    input_target[:, 0] = 1.0
                else:
                    input_target = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)
            elif self.label_type == SyntheticDataLabelType.RANDOM_INT:
                # use a dummy value for max int value
                dummy_max = 10
                input_target = torch.randint(0, dummy_max, (self.batch_size, *self.label_shape), device=self.device)
            else:
                raise ValueError(f"Unsupported label type {self.data_type}")

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

    batch_size: int = hp.required("Number of samples in a batch")
    data_shape: List[int] = hp.required("Shape of the data tensor.")
    batches_in_dataset: int = hp.optional("The number of batches in the dataset to emulate.", default=1)
    data_type: SyntheticDataType = hp.optional("Type of synthetic data to create.", default=SyntheticDataType.GAUSSIAN)
    label_type: SyntheticDataLabelType = hp.optional("Type of synthetic label to create.",
                                                     default=SyntheticDataLabelType.CLASSIFICATION)
    num_classes: int = hp.optional(
        "Number of classes. Required if label_type is SyntheticDataLabelType.CLASSIFICATION.", default=2)
    one_hot: bool = hp.optional(
        "Whether to use one-hot encoding. Required if label_type is SyntheticDataLabelType.CLASSIFICATION.",
        default=False)
    label_shape: List[int] = hp.optional(
        "Shape of the label tensor. Required if label_type is SyntheticDataLabelType.RANDOM_INT.",
        default_factory=lambda: [1])
    device: str = hp.optional(
        "Device to store the sample pool. "
        "Set to `cuda` to store samples on the GPU and eliminate PCI-e bandwidth with the dataloader. "
        "Set to `cpu` to move data between host memory and the gpu on every batch. ",
        default="cpu")
    memory_format: MemoryFormat = hp.optional("Memory format for the samples.", default=MemoryFormat.CONTIGUOUS_FORMAT)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

    def validate(self):
        super().validate()

        _validate_label_inputs(label_type=self.label_type,
                               num_classes=self.num_classes,
                               one_hot=self.one_hot,
                               label_shape=self.label_shape)

    def initialize_object(self) -> DataloaderSpec:
        return DataloaderSpec(
            SyntheticDataset(
                batch_size=self.batch_size,
                data_shape=self.data_shape,
                batches_in_dataset=self.batches_in_dataset,
                data_type=self.data_type,
                label_type=self.label_type,
                num_classes=self.num_classes,
                one_hot=self.one_hot,
                label_shape=self.label_shape,
                device=self.device,
                memory_format=self.memory_format,
            ),
            drop_last=self.drop_last,
            shuffle=False,
        )


def _validate_label_inputs(label_type: SyntheticDataLabelType, num_classes: Optional[int], one_hot: Optional[bool],
                           label_shape: Optional[Sequence[int]]):
    if label_type == SyntheticDataLabelType.CLASSIFICATION and \
        (num_classes is None or num_classes <= 0 or one_hot is None):
        raise ValueError("label_type classification requires num_classes > 0 and one_hot to be specified")
    if label_type == SyntheticDataLabelType.RANDOM_INT and label_shape is None:
        raise ValueError("label_type random_int requires label_shape to be specified")
