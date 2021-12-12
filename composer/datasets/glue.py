import logging
from dataclasses import dataclass
from multiprocessing import cpu_count

import numpy as np
import transformers
import yahp as hp
from tqdm import tqdm

from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.datasets.lm_datasets import _split_dict_fn
from composer.utils import ddp

log = logging.getLogger(__name__)


@dataclass
class GLUEHparams(DatasetHparams):
    task: str = hp.optional("The GLUE task to train on.", default=None)
    tokenizer_name: str = hp.optional("The name of the tokenizer to preprocess text with.", default=None)
    split: str = hp.optional("Whether to use 'train', 'validation' or 'test' split.", default=None)
    max_seq_length: int = hp.optional(
        default=256, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

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
            "wnli": ("sentence1", "sentence2"),
        }

        if (self.max_seq_length % 8) != 0:
            raise ValueError("For best performance, please ensure that sequence lengths are a multiple of eight.")

        if self.task not in self.task_to_keys.keys():
            raise ValueError("The task must be a valid GLUE task, optiosn are {' ,'.join(self.task_to_keys.keys())}.")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataloaderSpec:
        # TODO (Moin): I think this code is copied verbatim in a few different places. Move this into a function.
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)

        self.dataset = datasets.load_dataset("glue", self.task, split=self.split)
        print(f"Loading {self.task.upper()}...")

        n_cpus = cpu_count()
        log.info(f"Starting tokenization step by preprocessing over {n_cpus} threads!")
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
        sampler = ddp.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return DataloaderSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=data_collator,
        ),
                              split_fn=_split_dict_fn)
