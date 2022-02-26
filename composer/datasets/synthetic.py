# Copyright 2021 MosaicML. All Rights Reserved.

import json
import random
import string
import textwrap
from os.path import join
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence, Union

import torch
import torch.utils.data
from PIL import Image
from torchvision.datasets import VisionDataset

from composer.core.types import MemoryFormat
from composer.utils.string_enum import StringEnum

if TYPE_CHECKING:
    import tokenizers.decoders as decoders
    import tokenizers.models as tokenizers_models
    import tokenizers.normalizers as normalizers
    import tokenizers.pre_tokenizers as pre_tokenizers
    import tokenizers.trainers as tokenizers_trainer
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizer


class SyntheticDataType(StringEnum):
    GAUSSIAN = "gaussian"
    SEPARABLE = "separable"


class SyntheticDataLabelType(StringEnum):
    CLASSIFICATION_INT = "classification_int"
    CLASSIFICATION_ONE_HOT = "classification_one_hot"


class SyntheticTokenizerParams(NamedTuple):
    try:
        import tokenizers.decoders as decoders
        import tokenizers.models as tokenizers_models
        import tokenizers.normalizers as normalizers
        import tokenizers.pre_tokenizers as pre_tokenizers
    except ImportError as e:
        decoders = object
        tokenizers_models = object
        normalizers = object
        pre_tokenizers = object

    tokenizer_model: tokenizers_models.Model
    normalizer: normalizers.Normalizer
    pre_tokenizer: pre_tokenizers.PreTokenizer
    decoder: decoders.Decoder
    initial_alphabet: list
    special_tokens: list
    pad_token: str
    trainer_cls: type
    tokenizer_cls: type


def generate_bert_tokenizer_params(dataset) -> SyntheticTokenizerParams:
    try:
        import tokenizers.decoders as decoders
        import tokenizers.models as tokenizers_models
        import tokenizers.normalizers as normalizers
        import tokenizers.pre_tokenizers as pre_tokenizers
        import tokenizers.trainers as tokenizers_trainer
        from transformers import BertTokenizer
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
            if using pip or `conda install -c conda-forge transformers tokenizers` if using Anaconda.""")) from e

    unk_token = "[UNK]"
    pad_token = "[PAD]"

    initial_alphabet = "".join([i for i in dataset])
    initial_alphabet = list(set(initial_alphabet))

    return SyntheticTokenizerParams(
        tokenizer_model=tokenizers_models.WordPiece(unk_token=unk_token),  # type: ignore
        normalizer=normalizers.BertNormalizer(),
        pre_tokenizer=pre_tokenizers.BertPreTokenizer(),
        decoder=decoders.WordPiece(),
        initial_alphabet=initial_alphabet,
        special_tokens=[pad_token, unk_token, "[SEP]", "[CLS]", "[MASK]"],
        pad_token=pad_token,
        trainer_cls=tokenizers_trainer.WordPieceTrainer,
        tokenizer_cls=BertTokenizer,
    )


def generate_gpt2_tokenizer_params() -> SyntheticTokenizerParams:
    try:
        import tokenizers.decoders as decoders
        import tokenizers.models as tokenizers_models
        import tokenizers.normalizers as normalizers
        import tokenizers.pre_tokenizers as pre_tokenizers
        import tokenizers.trainers as tokenizers_trainer
        from transformers import GPT2Tokenizer
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
            if using pip or `conda install -c conda-forge tokenizers` if using Anaconda.""")) from e

    unk_token = None
    pad_token = "<pad>"

    return SyntheticTokenizerParams(
        tokenizer_model=tokenizers_models.BPE(unk_token=unk_token),
        normalizer=normalizers.Lowercase(),
        pre_tokenizer=pre_tokenizers.ByteLevel(),
        decoder=decoders.ByteLevel(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[pad_token, "<endoftext>"],
        pad_token=pad_token,
        trainer_cls=tokenizers_trainer.BpeTrainer,
        tokenizer_cls=GPT2Tokenizer,
    )


def generate_synthetic_tokenizer(tokenizer_family: str, dataset=None, vocab_size=256) -> PreTrainedTokenizer:
    try:
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizer
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
            if using pip or `conda install -c conda-forge transformers tokenizers` if using Anaconda.""")) from e

    # generate a synthetic dataset with reasonable defaults is none is provided
    if dataset is None:
        num_samples = 100
        chars_per_sample = 128
        column_names = ['sentence']
        dataset = SyntheticHFDataset(num_samples=num_samples,
                                     chars_per_sample=chars_per_sample,
                                     column_names=column_names).generate_dataset()

    # flatten dataset to get a simple stream of text for the tokenizer
    dataset = [dataset[key] for key in dataset.column_names if key != 'idx']
    dataset = [i for sublist in dataset for i in sublist]

    if tokenizer_family == "bert":
        tokenizer_params = generate_bert_tokenizer_params(dataset)
    elif tokenizer_family == "gpt2":
        tokenizer_params = generate_gpt2_tokenizer_params()
    else:
        raise ValueError(f"Synthetic tokenizers for tokenizer family {tokenizer_family} are currently unsupported.")

    tokenizer = Tokenizer(tokenizer_params.tokenizer_model)
    tokenizer.enable_padding(direction="right",
                             pad_id=0,
                             pad_type_id=0,
                             pad_token=tokenizer_params.pad_token,
                             pad_to_multiple_of=8)
    tokenizer.normalizer = tokenizer_params.normalizer  # type: ignore
    tokenizer.pre_tokenizer = tokenizer_params.pre_tokenizer  # type: ignore
    tokenizer.decoder = tokenizer_params.decoder  # type: ignore
    tokenizer_trainer = tokenizer_params.trainer_cls(
        vocab_size=vocab_size,
        initial_alphabet=tokenizer_params.initial_alphabet,
        special_tokens=tokenizer_params.special_tokens,
    )
    tokenizer.train_from_iterator(dataset, trainer=tokenizer_trainer)

    # save the tokenizer config
    tmp_tokenizer_dir = mkdtemp()
    tmp_tokenizer_file = join(tmp_tokenizer_dir, "tokenizer.json")
    tokenizer.save(tmp_tokenizer_file)
    print("Temporary directory:", tmp_tokenizer_dir)

    # save the vocabulary and potential merges file
    tokenizer_params.tokenizer_model.save(tmp_tokenizer_dir)  # type: ignore

    # the .from_pretrained method doesn't load our padding for some reason, so we save it as a special kwarg
    tmp_tokenizer_config = join(tmp_tokenizer_dir, "tokenizer_config.json")
    with open(tmp_tokenizer_config, "w") as f:
        json.dump({"pad_token": tokenizer_params.pad_token}, f)

    # instantiate the new tokenizer
    assert isinstance(tokenizer_params.tokenizer_cls, PreTrainedTokenizer)
    tokenizer = tokenizer_params.tokenizer_cls.from_pretrained(tmp_tokenizer_dir)

    return tokenizer


class SyntheticHFDataset:
    """Creates a synthetic HF dataset and passes it to the preprocessing scripts."""

    def __init__(self, num_samples: int, chars_per_sample: int, column_names: list):
        if column_names is None or len(column_names) == 0:
            raise ValueError("There must be at least one column name provided for the final dataset.")
        self.num_samples = num_samples
        self.chars_per_sample = chars_per_sample
        self.column_names = column_names

    def generate_dataset(self):
        try:
            import datasets
        except ImportError as e:
            raise ImportError(
                'Composer was installed without NLP support. To use NLP with Composer, run: `pip install mosaicml[nlp]`.'
            ) from e

        data = {}
        for column_name in self.column_names:
            data[column_name] = [self.generate_sample() for _ in range(self.num_samples)]
        data['idx'] = list(range(self.num_samples))

        hf_synthetic_dataset = datasets.Dataset.from_dict(data)
        return hf_synthetic_dataset

    def generate_sample(self):
        MIN_WORD_LENGTH = 3
        MAX_WORD_LENGTH = 10
        character_set = {
            "letters": {
                "weight": 10,
                "choices": string.ascii_letters
            },
            "digits": {
                "weight": 5,
                "choices": string.digits
            },
            "punctuation": {
                "weight": 1,
                "choices": string.punctuation
            }
        }
        valid_chars = ''.join([(i['choices'] * i['weight']) for i in character_set.values()])

        sample = ''
        while len(sample) < self.chars_per_sample:
            sample_len = random.randint(MIN_WORD_LENGTH, MAX_WORD_LENGTH)
            sample += ''.join([random.choice(valid_chars) for _ in range(sample_len)])
            sample += ' '
        sample = sample[:self.chars_per_sample]
        return sample


class SyntheticBatchPairDataset(torch.utils.data.Dataset):
    """Emulates a dataset of provided size and shape.

    Args:
        total_dataset_size (int): The total size of the dataset to emulate.
        data_shape (List[int]): Shape of the tensor for input samples.
        num_unique_samples_to_create (int): The number of unique samples to allocate memory for.
        data_type (str or SyntheticDataType, optional), Type of synthetic data to create.
        label_type (str or SyntheticDataLabelType, optional), Type of synthetic data to create.
        num_classes (int, optional): Number of classes to use. Required if
            ``SyntheticDataLabelType`` is ``CLASSIFICATION_INT`` or``CLASSIFICATION_ONE_HOT``. Otherwise, should be ``None``.
        label_shape (List[int]): Shape of the tensor for each sample label.
        device (str): Device to store the sample pool. Set to ``cuda`` to store samples
            on the GPU and eliminate PCI-e bandwidth with the dataloader. Set to `cpu`
            to move data between host memory and the gpu on every batch.
        memory_format (MemoryFormat, optional): Memory format for the sample pool.
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
                 device: str = "cpu",
                 memory_format: Union[str, MemoryFormat] = MemoryFormat.CONTIGUOUS_FORMAT,
                 transform: Optional[Callable] = None):
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
                raise ValueError("classification label_types require num_classes > 0")

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


class SyntheticPILDataset(VisionDataset):
    """Similar to :class:`SyntheticBatchPairDataset`, but yields samples of type :class:`~Image.Image` and supports
    dataset transformations.

    Args:
        total_dataset_size (int): The total size of the dataset to emulate.
        data_shape (List[int]): Shape of the image for input samples. Default = [64, 64]
        num_unique_samples_to_create (int): The number of unique samples to allocate memory for.
        data_type (str or SyntheticDataType, optional), Type of synthetic data to create.
        label_type (str or SyntheticDataLabelType, optional), Type of synthetic data to create.
        num_classes (int, optional): Number of classes to use. Required if
            ``SyntheticDataLabelType`` is ``CLASSIFICATION_INT`` or
            ``CLASSIFICATION_ONE_HOT``. Otherwise, should be ``None``.
        label_shape (List[int]): Shape of the tensor for each sample label.
        transform (Callable): Dataset transforms
    """

    def __init__(self,
                 *,
                 total_dataset_size: int,
                 data_shape: Sequence[int] = (64, 64, 3),
                 num_unique_samples_to_create: int = 100,
                 data_type: Union[str, SyntheticDataType] = SyntheticDataType.GAUSSIAN,
                 label_type: Union[str, SyntheticDataLabelType] = SyntheticDataLabelType.CLASSIFICATION_INT,
                 num_classes: Optional[int] = None,
                 label_shape: Optional[Sequence[int]] = None,
                 transform: Optional[Callable] = None):
        super().__init__(root="", transform=transform)
        self._dataset = SyntheticBatchPairDataset(
            total_dataset_size=total_dataset_size,
            data_shape=data_shape,
            data_type=data_type,
            num_unique_samples_to_create=num_unique_samples_to_create,
            label_type=label_type,
            num_classes=num_classes,
            label_shape=label_shape,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        input_data, target = self._dataset[idx]

        input_data = input_data.numpy()

        # Shift and scale to be [0, 255]
        input_data = (input_data - input_data.min())
        input_data = (input_data * (255 / input_data.max())).astype("uint8")

        sample = Image.fromarray(input_data)
        if self.transform is not None:
            return self.transform(sample), target
        else:
            return sample, target
