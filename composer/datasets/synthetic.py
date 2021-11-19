# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.utils.data
import yahp as hp

from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.utils.string_enum import StringEnum


class SyntheticDataType(StringEnum):
    GAUSSIAN = "gaussian"
    SEPARABLE = "separable"


class SyntheticDataLabelType(StringEnum):
    CLASSIFICATION_INT = "classification_int"
    CLASSIFICATION_ONE_HOT = "classification_one_hot"
    RANDOM_INT = "random_int"


class MemoryFormat(StringEnum):
    CONTIGUOUS_FORMAT = "contiguous_format"
    CHANNELS_LAST = "channels_last"
    CHANNELS_LAST_3D = "channels_last_3d"
    PRESERVE_FORMAT = "preserve_format"


class SyntheticDataset(torch.utils.data.Dataset):
    """Emulates a dataset of provided size and shape.

    Args:
        total_dataset_size (int): The total size of the dataset to emulate.
        data_shape (List[int]): Shape of the tensor for input samples.
        num_unique_samples_to_create (int): The number of unique samples to allocate memory for.
        data_type (SyntheticDataType, optional), Type of synthetic data to create.
        label_type (SyntheticDataLabelType, optional), Type of synthetic data to create.
        num_classes (int, optional): Number of classes to use. Required if `SyntheticDataLabelType`
            is `CLASSIFICATION_INT` or `CLASSIFICATION_ONE_HOT`. Otherwise, should be `None`.
        label_shape (List[int]): Shape of the tensor for each sample label.
        device (str): Device to store the sample pool. Set to `cuda` to store samples
            on the GPU and eliminate PCI-e bandwidth with the dataloader. Set to `cpu`
            to move data between host memory and the gpu on every batch.
        memory_format (MemoryFormat, optional): Memory format for the sample pool.
        drop_last (bool): Whether to drop the last samples for the last batch.
        shuffle (bool): Whether to shuffle the dataset for each epoch.
    """

    def __init__(self,
                 *,
                 total_dataset_size: int,
                 data_shape: Sequence[int],
                 num_unique_samples_to_create: int = 100,
                 data_type: SyntheticDataType = SyntheticDataType.GAUSSIAN,
                 label_type: SyntheticDataLabelType = SyntheticDataLabelType.CLASSIFICATION_INT,
                 num_classes: Optional[int] = None,
                 label_shape: Optional[Sequence[int]] = None,
                 device: str = "cpu",
                 memory_format: Union[str, MemoryFormat] = MemoryFormat.CONTIGUOUS_FORMAT,
                 transform: Optional[Callable] = None):
        self.total_dataset_size = total_dataset_size
        self.data_shape = data_shape
        self.num_unique_samples_to_create = num_unique_samples_to_create
        self.data_type = data_type
        self.label_type = label_type
        self.num_classes = num_classes
        self.label_shape = label_shape
        self.device = device
        self.memory_format = getattr(torch, MemoryFormat(memory_format).value)
        self.transform = transform

        _validate_label_inputs(label_type=self.label_type, num_classes=self.num_classes, label_shape=self.label_shape)

        # The synthetic data
        self.input_data = None
        self.input_target = None

    def __len__(self) -> int:
        return self.total_dataset_size

    def __getitem__(self, idx: int):
        idx = idx % self.num_unique_samples_to_create
        if self.input_data is None:
            # Generating data on the first call to __getitem__ so that data is stored on the correct gpu,
            # after DeviceSingleGPU calls torch.cuda.set_device
            # This does mean that the first batch will be slower
            # generating samples so all values for the sample are the sample index
            # e.g. all(input_data[1] == 1). Helps with debugging.
            assert self.input_target is None
            input_data = torch.randn(self.num_unique_samples_to_create, *self.data_shape, device=self.device)

            input_data = torch.clone(input_data)  # allocate actual memory
            input_data = input_data.contiguous(memory_format=self.memory_format)

            if self.label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
                assert self.num_classes is not None
                input_target = torch.zeros((self.num_unique_samples_to_create, self.num_classes), device=self.device)
                input_target[:, 0] = 1.0
            elif self.label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
                assert self.num_classes is not None
                input_target = torch.randint(0,
                                             self.num_classes, (self.num_unique_samples_to_create,),
                                             device=self.device)
            elif self.label_type == SyntheticDataLabelType.RANDOM_INT:
                assert self.label_shape is not None
                # use a dummy value for max int value
                dummy_max = 10
                input_target = torch.randint(0,
                                             dummy_max, (self.num_unique_samples_to_create, *self.label_shape),
                                             device=self.device)
            else:
                raise ValueError(f"Unsupported label type {self.data_type}")

            # If separable, force the positive examples to have a higher mean than the negative examples
            if self.data_type == SyntheticDataType.SEPARABLE:
                assert self.label_type == SyntheticDataLabelType.CLASSIFICATION_INT, \
                    "SyntheticDataType.SEPARABLE requires integer classes."
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

    See :class:`~composer.datasetes.synthetic.SyntheticDataset`
    """

    total_dataset_size: int = hp.required("The total size of the dataset to emulate.")
    data_shape: List[int] = hp.required("Shape of the data tensor.")
    num_unique_samples_to_create: int = hp.optional("The number of unique samples to allocate memory for.", default=100)
    data_type: SyntheticDataType = hp.optional("Type of synthetic data to create.", default=SyntheticDataType.GAUSSIAN)
    label_type: SyntheticDataLabelType = hp.optional("Type of synthetic label to create.",
                                                     default=SyntheticDataLabelType.CLASSIFICATION_INT)
    num_classes: Optional[int] = hp.optional(
        "Number of classes. Required if label_type is SyntheticDataLabelType.CLASSIFICATION_INT or "
        "SyntheticDataLabelType.CLASSIFICATION_ONE_HOT.",
        default=2)
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

        _validate_label_inputs(label_type=self.label_type, num_classes=self.num_classes, label_shape=self.label_shape)

    def initialize_object(self) -> DataloaderSpec:
        return DataloaderSpec(
            SyntheticDataset(
                total_dataset_size=self.total_dataset_size,
                data_shape=self.data_shape,
                num_unique_samples_to_create=self.num_unique_samples_to_create,
                data_type=self.data_type,
                label_type=self.label_type,
                num_classes=self.num_classes,
                label_shape=self.label_shape,
                device=self.device,
                memory_format=self.memory_format,
            ),
            drop_last=self.drop_last,
            shuffle=False,
        )


def _validate_label_inputs(label_type: SyntheticDataLabelType, num_classes: Optional[int],
                           label_shape: Optional[Sequence[int]]):
    if label_type == SyntheticDataLabelType.CLASSIFICATION_INT or label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
        if num_classes is None or num_classes <= 0:
            raise ValueError("classification label_types require num_classes > 0")
    if label_type == SyntheticDataLabelType.RANDOM_INT and label_shape is None:
        raise ValueError("label_type random_int requires label_shape to be specified")
