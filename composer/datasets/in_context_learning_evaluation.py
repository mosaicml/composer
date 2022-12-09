# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from composer.core import DataSpec


class PerplexityTaskDataset(IterableDataset):

    def __init__(
        self,
        dataset_uri: str,
        tokenizer: str,
    ):
        dataset = load_dataset('json', data_files=dataset_uri, split='train', streaming=True)
        self.encoded_dataset = dataset.map(lambda examples: {
            'continuation': tokenizer(examples['continuation']),
            'context': tokenizer(examples['context']),
        })

    def __iter__(self):
        for example in self.encoded_dataset:
            yield example


def get_perplexity_task_dataloader(dataset_uri, tokenizer, batch_size):
    dataset = PerplexityTaskDataset(dataset_uri, tokenizer)
    return DataSpec(
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=None,
            collate_fn=None,
        ),
        device_transforms=None,
    )
