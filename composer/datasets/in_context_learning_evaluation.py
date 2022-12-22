# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import inspect
import textwrap

import torch
from torch.utils.data import DataLoader, Dataset

from composer.core import DataSpec
from composer.utils import MissingConditionalImportError, dist, get_file

if TYPE_CHECKING:
    import transformers

__all__ = ['InContextLearningLMTaskDataset', 'get_lm_task_dataloader']


class InContextLearningLMTaskDataset(Dataset):
    """A dataset that construct batches for in-context learning language modeling evaluation

    Args:
        dataset_uri (str): Either a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used to transform data into batches
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The sequence length expected by the model
        eos_tok_id (int): The special token reserved for padding the ends of batches
        destination_path (str): Temporary path to store downloaded datasets
    """

    def __init__(
        self,
        dataset_uri: str,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        max_seq_len: int,
        eos_tok_id: int,
        destination_path: str = 'icl_lm_task.json',
    ):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='datasets',
                                                conda_channel='conda-forge') from e

        get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        self.encoded_dataset = list(
            dataset.map(lambda examples: {
                'continuation': tokenizer(examples['continuation']),
                'context': tokenizer(examples['context']),
            }))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_tok_id = eos_tok_id

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

    def collate_fn(self, data):
        inputs = []
        continuation_indices = []
        for data_pair in data:
            context, continuation = data_pair['context'], data_pair['continuation']

            context_enc = context['input_ids']
            continuation_enc = continuation['input_ids']
            continuation_span = torch.tensor(range(len(context_enc), len(context_enc) + len(continuation_enc)))

            inp = torch.tensor(
                (context_enc + continuation_enc
                )[-(self.max_seq_len + 1):],  # trim from the left if context + continuation are larger than max_seq_len
                dtype=torch.long,
            )
            (inp_len,) = inp.shape

            # pad length from seq to padding_length
            inp = torch.cat(
                [
                    inp,  # [seq]
                    torch.LongTensor((self.max_seq_len - inp_len) * [self.eos_tok_id]),
                ],
                dim=0,
            )

            inputs.append(inp)
            continuation_indices.append(continuation_span)

        batch = {
            'input_ids': torch.stack(inputs),
            'continuation_indices': continuation_indices,
            'mode': 'lm_task',
            'labels': torch.stack(inputs),
        }

        batch['attention_mask'] = ~(batch['input_ids'] == self.eos_tok_id)
        return batch

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0]

    def update_metric(self, metric, batch, output_logits, labels):
        metric.update(batch, output_logits, labels)


def get_lm_task_dataloader(dataset_uri: str, tokenizer: Union[transformers.PreTrainedTokenizer,
                                                              transformers.PreTrainedTokenizerFast], batch_size: int,
                           max_seq_len: int, eos_tok_id: int) -> DataSpec:
    """This constructs a dataloader capable of evaluating LLMs on in-context learning language modeling tasks, for example LAMBADA. An example usage is below:

    >>> dl = get_lm_task_dataloader(dataset_uri, tokenizer, 2, max_seq_len=2048, eos_tok_id=tokenizer.eos_token_id)
    >>> eval_evaluator = Evaluator(
       ...     label="lambada",
       ...     dataloader=dl,
       ...     metric_names=['InContextLearningLMAccuracy']
       ... )
    >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_evaluator,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ... )

    Args:
        dataset_uri (str): Either a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used to transform data into batches
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The sequence length expected by the model
        eos_tok_id (int): The special token reserved for padding the ends of batches

    Returns:
        DataLoader: A dataloader used for performing in-context learning evaluation on the dataset provided.
    """
    dataset = InContextLearningLMTaskDataset(dataset_uri, tokenizer, max_seq_len, eos_tok_id)
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=True)
    return DataSpec(DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
    ),
                    get_num_samples_in_batch=dataset.get_num_samples_in_batch)
