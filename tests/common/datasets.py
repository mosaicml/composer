# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Sequence

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.datasets import VisionDataset

from composer.utils import dist
from tests.common.models import configure_tiny_bert_tokenizer, configure_tiny_gpt2_tokenizer


class ParityDataset(Dataset):
    """A dataset of numbers where the output is the parity.

    Args:
        size (int): number of samples (default: 100)
    """

    def __init__(self, size: int = 100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return torch.tensor(index, dtype=torch.float32), torch.tensor(index % 2)


class InfiniteClassificationDataset(IterableDataset):
    """Classification dataset that never ends.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), num_classes: int = 2):
        self.shape = shape
        self.num_classes = num_classes

    def __iter__(self):
        while True:
            yield torch.randn(*self.shape), torch.randint(0, self.num_classes, size=(1,))[0]


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        return self.x[index], self.y[index]


class RandomImageDataset(VisionDataset):
    """ Image Classification dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (32, 32, 3)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self, shape: Sequence[int] = (3, 32, 32), size: int = 100, num_classes: int = 2, is_PIL: bool = False):
        self.is_PIL = is_PIL
        if is_PIL:  # PIL expects HWC
            shape = (shape[1], shape[2], shape[0])
        self.shape = shape
        self.num_classes = num_classes

        self.size = size
        self.x = None
        self.y = None

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype('uint8')
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y


class RandomSegmentationDataset(VisionDataset):
    """ Image Segmentation dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (32, 32, 3)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self, shape: Sequence[int] = (3, 32, 32), size: int = 100, num_classes: int = 2, is_PIL: bool = False):
        self.is_PIL = is_PIL
        if is_PIL:  # PIL expects HWC
            shape = (shape[1], shape[2], shape[0])
        self.shape = shape
        self.num_classes = num_classes

        self.size = size
        self.x = None
        self.y = None

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            mask_shape = self.shape[:2] if self.is_PIL else self.shape[1:]
            self.y = torch.randint(0, self.num_classes, size=(self.size, *mask_shape))
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype('uint8')
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y


class RandomTextClassificationDataset(Dataset):
    """ Text classification dataset with values (just input token ids) drawn uniformly
    Args:
        vocab_size (int): vocab size to use (default: 10)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        sequence_length (int): sequence length to use, all sequences will be of this length with no padding (default: 8)
        use_keys: (bool): whether to return the item in a dictionary with keys for input and output
    """

    def __init__(self,
                 size: int = 100,
                 vocab_size: int = 10,
                 sequence_length: int = 8,
                 num_classes: int = 2,
                 use_keys: bool = False):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.use_keys = use_keys

        self.input_key = 'input_ids'
        self.label_key = 'labels'

        self.size = size
        self.x = None
        self.y = None

        super().__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randint(low=0, high=self.vocab_size, size=(self.size, self.sequence_length))
        if self.y is None:
            self.y = torch.randint(low=0, high=self.num_classes, size=(self.size,))

        x = self.x[index]
        y = self.y[index]

        if self.use_keys:
            return {'input_ids': x, 'labels': y}
        else:
            return x, y


class RandomTextRegressionDataset(Dataset):
    """ Text Regression dataset with values (just input token ids) drawn uniformly
    Args:
        vocab_size (int): vocab size to use (default: 10)
        size (int): number of samples (default: 100)
        sequence_length (int): sequence length to use, all sequences will be of this length with no padding (default: 8)
        use_keys: (bool): whether to return the item in a dictionary with keys for input and output
    """

    def __init__(self, size: int = 100, vocab_size: int = 10, sequence_length: int = 8, use_keys: bool = False):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.use_keys = use_keys

        self.input_key = 'input_ids'
        self.label_key = 'labels'

        self.size = size
        self.x = None
        self.y = None

        super().__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randint(low=0, high=self.vocab_size, size=(self.size, self.sequence_length))
        if self.y is None:
            self.y = torch.rand(size=(self.size,))

        x = self.x[index]
        y = self.y[index]
        if self.use_keys:
            return {'input_ids': x, 'labels': y}
        else:
            return x, y


class RandomTextLMDataset(Dataset):
    """ Text LM dataset with values (just input token ids) drawn uniformly
    Args:
        vocab_size (int): vocab size to use (default: 10)
        size (int): number of samples (default: 100)
        sequence_length (int): sequence length to use, all sequences will be of this length with no padding (default: 8)
        use_keys: (bool): whether to return the item in a dictionary with keys for input and output
    """

    def __init__(self,
                 size: int = 100,
                 vocab_size: int = 10,
                 sequence_length: int = 8,
                 use_keys: bool = False,
                 use_token_type_ids: bool = True,
                 conditional_generation: bool = False,
                 causal_lm: bool = False,
                 pad_token_id: Optional[int] = None):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.use_keys = use_keys
        self.use_token_type_ids = use_token_type_ids
        self.conditional_generation = conditional_generation
        self.causal_lm = causal_lm
        self.pad_token_id = pad_token_id

        self.input_key = 'input_ids'

        self.size = size
        self.x = None
        self.y = None

        super().__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randint(low=0, high=self.vocab_size, size=(self.size, self.sequence_length))
            if self.pad_token_id is not None:
                mask = torch.randint(low=0, high=2, size=(self.size, self.sequence_length // 2)).bool()
                self.x[:, :self.sequence_length // 2][mask] = self.pad_token_id
            if self.conditional_generation:
                self.y = torch.randint(low=0, high=self.vocab_size, size=(self.size, 2 * self.sequence_length))
            if self.causal_lm:
                self.y = torch.randint(low=0, high=self.vocab_size, size=(self.size, self.sequence_length))

        x = self.x[index]

        if self.use_keys:
            output = {'input_ids': x}
            if self.use_token_type_ids:
                output['token_type_ids'] = torch.zeros_like(x)
            if self.y is not None:
                output['labels'] = self.y[index]
            return output
        else:
            return x if self.y is None else (x, self.y[index])


class SimpleDataset(Dataset):

    def __init__(self, size: int = 256, batch_size: int = 256, feature_size: int = 1, num_classes: int = 2):
        self.size = size
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size * self.batch_size, self.feature_size)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size * self.batch_size,), dtype=torch.long)
        return self.x[index * self.batch_size:(index + 1) *
                      self.batch_size], self.y[index * self.batch_size:(index + 1) * self.batch_size]


class SyntheticDataType(StringEnum):
    """Defines the distribution of the synthetic data.
    Attributes:
        GAUSSIAN: Standard Gaussian distribution.
        SEPARABLE: Gaussian distributed, but classes will be mean-shifted for
            separability.
    """

    GAUSSIAN = 'gaussian'
    SEPARABLE = 'separable'


class SyntheticBatchPairDataset(torch.utils.data.Dataset):
    """Emulates a dataset of provided size and shape.
    Args:
        total_dataset_size (int): The total size of the dataset to emulate.
        data_shape (List[int]): Shape of the tensor for input samples.
        num_unique_samples_to_create (int): The number of unique samples to allocate memory for.
        data_type (str or SyntheticDataType, optional), Type of synthetic data to create.
            Default: ``SyntheticDataType.GAUSSIAN``.
        label_type (str or SyntheticDataLabelType, optional), Type of synthetic data to
            create. Default: ``SyntheticDataLabelType.CLASSIFICATION_INT``.
        num_classes (int, optional): Number of classes to use. Required if
            ``SyntheticDataLabelType`` is ``CLASSIFICATION_INT``
            or``CLASSIFICATION_ONE_HOT``. Default: ``None``.
        label_shape (List[int], optional): Shape of the tensor for each sample label.
            Default: ``None``.
        device (str): Device to store the sample pool. Set to ``'cuda'`` to store samples
            on the GPU and eliminate PCI-e bandwidth with the dataloader. Set to ``'cpu'``
            to move data between host memory and the gpu on every batch. Default:
            ``'cpu'``.
        memory_format (:class:`composer.core.MemoryFormat`, optional): Memory format for the sample pool.
            Default: `MemoryFormat.CONTIGUOUS_FORMAT`.
        transform (Callable, optional): Transform(s) to apply to data. Default: ``None``.
    """

    def __init__(self,
                 *,
                 total_dataset_size: int,
                 data_shape: Sequence[int],
                 num_unique_samples_to_create: int = 100,
                 data_type: Union[str, SyntheticDataType] = SyntheticDataType.GAUSSIAN,
                 label_type: Union[str, SyntheticDataLabelType] = SyntheticDataLabelType.CLASSIFICATION_INT,
                 num_classes: Optional[int] = None,
                 label_shape: Optional[Sequence[int]] = None,
                 device: str = 'cpu',
                 memory_format: Union[str, MemoryFormat] = MemoryFormat.CONTIGUOUS_FORMAT,
                 transform: Optional[Callable] = None):
        warnings.warn(DeprecationWarning('SyntheticBatchPairDataset is deprecated and will be removed in v0.18'))

        self.total_dataset_size = total_dataset_size
        self.data_shape = data_shape
        self.num_unique_samples_to_create = num_unique_samples_to_create
        self.data_type = SyntheticDataType(data_type)
        self.label_type = SyntheticDataLabelType(label_type)
        self.num_classes = num_classes
        self.label_shape = label_shape
        self.device = device
        self.memory_format = MemoryFormat(memory_format)
        self.transform = transform

        self._validate_label_inputs(label_type=self.label_type,
                                    num_classes=self.num_classes,
                                    label_shape=self.label_shape)

        # The synthetic data
        self.input_data = None
        self.input_target = None

    def _validate_label_inputs(self, label_type: SyntheticDataLabelType, num_classes: Optional[int],
                               label_shape: Optional[Sequence[int]]):
        if label_type == SyntheticDataLabelType.CLASSIFICATION_INT or label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
            if num_classes is None or num_classes <= 0:
                raise ValueError('classification label_types require num_classes > 0')

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
            input_data = input_data.contiguous(memory_format=getattr(torch, self.memory_format.value))

            if self.label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
                assert self.num_classes is not None
                input_target = torch.zeros((self.num_unique_samples_to_create, self.num_classes), device=self.device)
                input_target[:, 0] = 1.0
            elif self.label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
                assert self.num_classes is not None
                if self.label_shape:
                    label_batch_shape = (self.num_unique_samples_to_create, *self.label_shape)
                else:
                    label_batch_shape = (self.num_unique_samples_to_create,)
                input_target = torch.randint(0, self.num_classes, label_batch_shape, device=self.device)
            else:
                raise ValueError(f'Unsupported label type {self.data_type}')

            # If separable, force the positive examples to have a higher mean than the negative examples
            if self.data_type == SyntheticDataType.SEPARABLE:
                assert self.label_type == SyntheticDataLabelType.CLASSIFICATION_INT, \
                    'SyntheticDataType.SEPARABLE requires integer classes.'
                assert torch.max(input_target) == 1 and torch.min(input_target) == 0, \
                    'SyntheticDataType.SEPARABLE only supports binary labels'
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


def dummy_transformer_classifier_batch(vocab_size=10, num_classes=2):
    sequence_length = 32
    size = 8
    batch_size = 8
    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    return next(iter(train_dataloader))


def dummy_tiny_bert_classification_batch(num_classes=2):
    vocab_size = 30522  # Match bert vocab size
    sequence_length = 4
    size = 8
    batch_size = 8

    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes,
                                                    use_keys=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    batch = next(iter(train_dataloader))
    return batch


def dummy_tiny_bert_lm_batch():
    vocab_size = 30522  # Match bert vocab size
    sequence_length = 4
    size = 8
    batch_size = 8

    train_dataset = RandomTextLMDataset(size=size,
                                        vocab_size=vocab_size,
                                        sequence_length=sequence_length,
                                        use_keys=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    batch = next(iter(train_dataloader))
    return batch


def dummy_hf_lm_dataloader(size: int, vocab_size: int, sequence_length: int, collate_fn=None):
    batch_size = 2

    dataset = RandomTextLMDataset(size=size, vocab_size=vocab_size, sequence_length=sequence_length, use_keys=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=dist.get_sampler(dataset), collate_fn=collate_fn)
    return dataloader


def dummy_bert_lm_dataloader(sequence_length=4, size=4):
    transformers = pytest.importorskip('transformers')
    tokenizer = configure_tiny_bert_tokenizer()
    collate_fn = transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                                 mlm=True,
                                                                                 mlm_probability=0.15)
    return dummy_hf_lm_dataloader(vocab_size=30522, sequence_length=sequence_length, size=size, collate_fn=collate_fn)


def dummy_gpt_lm_dataloader(sequence_length=4, size=4):
    transformers = pytest.importorskip('transformers')
    tokenizer = configure_tiny_gpt2_tokenizer()
    collate_fn = transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return dummy_hf_lm_dataloader(vocab_size=50257, sequence_length=sequence_length, size=size, collate_fn=collate_fn)


def dummy_text_classification_dataloader():
    dataset = RandomTextClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=4, sampler=dist.get_sampler(dataset))
    return dataloader
