# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from dataclasses import dataclass
from multiprocessing import cpu_count

import yahp as hp

from composer.core import DataSpec
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.datasets.lm_datasets import _split_dict_fn
from composer.utils import dist

log = logging.getLogger(__name__)


@dataclass
class GLUEHparams(DatasetHparams):
    """
    Sets up a generic GLUE dataset loader.

    Args:
        task (str): the GLUE task to train on, choose one from: CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, and STS-B.
        tokenizer_name (str): The name of the HuggingFace tokenizer to preprocess text with.
        split (str): Whether to use 'train', 'validation' or 'test' split.
        max_seq_length (int): Optionally, the ability to set a custom sequence length for the training dataset.
            Default: 256

    Returns:
        A :class:`~composer.core.DataSpec` object
    """

    task: str = hp.optional(
        "The GLUE task to train on, choose one from: CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, and STS-B.", default=None)
    tokenizer_name: str = hp.optional("The name of the tokenizer to preprocess text with.", default=None)
    split: str = hp.optional("Whether to use 'train', 'validation' or 'test' split.", default=None)
    max_seq_length: int = hp.optional(
        default=256, doc='Optionally, the ability to set a custom sequence length for the training dataset.')

    def validate(self):
        self.task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
        }

        if self.task not in self.task_to_keys.keys():
            raise ValueError(f"The task must be a valid GLUE task, options are {' ,'.join(self.task_to_keys.keys())}.")

        if (self.max_seq_length % 8) != 0:
            log.warning("For best hardware acceleration, it is recommended that sequence lengths be multiples of 8.")

        if self.tokenizer_name is None:
            raise ValueError("A tokenizer name must be specified to tokenize the dataset.")

        if self.split is None:
            raise ValueError("A dataset split must be specified.")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataSpec:
        # TODO (Moin): I think this code is copied verbatim in a few different places. Move this into a function.
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')

        self.validate()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)

        print(f"Loading {self.task.upper()} on rank ", dist.get_global_rank())
        download_config = datasets.utils.DownloadConfig(max_retries=10)
        self.dataset = datasets.load_dataset("glue", self.task, split=self.split, download_config=download_config)

        n_cpus = cpu_count() // dist.get_world_size()
        print(f"Starting tokenization step by preprocessing over {n_cpus} threads!")
        text_column_names = self.task_to_keys[self.task]

        def tokenize_function(inp):
            # truncates sentences to max_length or pads them to max_length

            first_half = inp[text_column_names[0]]
            second_half = inp[text_column_names[1]] if text_column_names[1] in inp else None
            return self.tokenizer(
                text=first_half,
                text_pair=second_half,
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
            )

        columns_to_remove = ["idx"] + [i for i in text_column_names if i is not None]
        assert isinstance(self.dataset, datasets.Dataset)
        dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=n_cpus,
            batch_size=1000,
            remove_columns=columns_to_remove,
            new_fingerprint=f"{self.task}-tokenization-{self.split}",
            load_from_cache_file=True,
        )

        data_collator = transformers.data.data_collator.default_data_collator
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return DataSpec(
            dataloader=dataloader_hparams.initialize_object(
                dataset=dataset,  #type: ignore (thirdparty)
                batch_size=batch_size,
                sampler=sampler,
                drop_last=self.drop_last,
                collate_fn=data_collator,
            ),
            split_batch=_split_dict_fn)
