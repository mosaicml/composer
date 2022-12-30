# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import random
import textwrap

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from composer.core import DataSpec
from composer.utils import dist, ensure_tuple
from composer.utils.file_helpers import get_file


class InContextLearningMultipleChoiceTaskDataset(Dataset):

    def __init__(
        self,
        dataset_uri: str,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        eos_tok_id: int,
        num_fewshot: int,
        preamble_string: str,
        example_delimiter: str,
        continuation_delimiter: str,
        destination_path: str = 'icl_mc_task.json',
    ):
        get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        self.samples = list(
            dataset.map(lambda examples: {
                'query': examples['query'],
                'choices': examples['choices'],
                'gold': examples['gold']
            }))
        self.num_choices = len(self.samples[0]['choices'])
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_tok_id = eos_tok_id
        self.encoded_dataset = self.prep_examples(num_fewshot, preamble_string, example_delimiter,
                                                  continuation_delimiter)

    def prep_examples(self, num_fewshot, preamble_string, example_delimiter, continuation_delimiter):
        examples = []
        for sample_idx in range(len(self.samples)):

            preamble = preamble_string
            if num_fewshot > 0:
                allowable_indices = list(range(len(self.samples)))
                allowable_indices.remove(sample_idx)
                fewshot_idxs = random.sample(allowable_indices, num_fewshot)

                for fewshot_idx in fewshot_idxs:
                    query, choices, gold_idx = self.samples[fewshot_idx]['query'], self.samples[fewshot_idx]['choices'], self.samples[fewshot_idx]['gold']
                    if len(preamble) > 0:
                        query = f'{example_delimiter}{query}'
                    preamble += f'{query}{continuation_delimiter}{choices[gold_idx]}'

            
            encoded_example = {}
            query, choices, gold_idx = self.samples[sample_idx]['query'], self.samples[sample_idx]['choices'], self.samples[sample_idx]['gold'],
            if len(preamble) > 0:
                query = f'{example_delimiter}{query}'

            choices = [f'{continuation_delimiter}{choice}' for choice in choices]
            encoded_example['query'] = self.tokenizer(query)
            encoded_example['choices'] = [self.tokenizer(choice) for choice in choices]
            encoded_example['preamble'] = self.tokenizer(preamble)
            encoded_example['gold_idx'] = gold_idx

            examples.append(encoded_example)

        return examples

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

    def collate_fn(self, data):
        inputs = []
        continuation_indices = []
        gold_idxs = []
        choice_groupings = []
        for data_pair in data:

            choice_start_idx = len(continuation_indices)        
            preamble, context, choices, gold_idx = (data_pair['preamble'], data_pair['query'], data_pair['choices'], data_pair['gold_idx'])

            for choice in choices:
                context_enc = preamble['input_ids'] + context['input_ids']
                continuation_enc = choice['input_ids']
                continuation_span = torch.tensor(range(len(context_enc), len(context_enc) + len(continuation_enc)))

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_seq_len + 1):],
                    dtype=torch.long,
                )
                (inplen,) = inp.shape

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.LongTensor((self.max_seq_len - inplen) * [self.eos_tok_id]),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inputs.append(inp)
                continuation_indices.append(continuation_span)
            
            gold_idxs.append(gold_idx)
            choice_end_idx = len(continuation_indices)  
            choice_groupings.append((choice_start_idx, choice_end_idx))
            

        return {
            'input_ids': torch.stack(inputs),
            'continuation_indices': continuation_indices,
            'mode': 'lm_task',
            'labels': torch.stack(inputs),
            'eos_tok_id': self.eos_tok_id,
            'gold_indices': gold_idxs,
            'choice_groupings': choice_groupings
        }

    def get_num_samples_in_batch(self, batch) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]

        dim0_sizes = []
        if isinstance(batch, (list, tuple)):
            for tensors in batch:
                for t in ensure_tuple(tensors):
                    if not hasattr(t, 'shape'):
                        raise ValueError('Unable to determine the batch size, batch contains'
                                         f'an element of type {type(t)}, which does not have a'
                                         'shape. Please use a DataSpec and provide a'
                                         '`get_num_samples_in_batch(your_batch) -> int` method.')
                    dim0_sizes.append(t.shape[0])
        elif isinstance(batch, dict):
            dim0_sizes = [t.shape[0] for t in batch.values() if isinstance(t, torch.Tensor)]

        if len(set(dim0_sizes)) == 1:
            return dim0_sizes[0]
        else:
            raise NotImplementedError(
                textwrap.dedent(f"""\
                    Cannot determine the batch size, as multiple Tensors of
                    different lengths were found in the batch: sizes in batch: {dim0_sizes}.
                    Please use a DataSpec and specify `get_num_samples_in_batch`."""))

    def update_metric(self, metric, batch, output_logits, labels):
        metric.update(batch, output_logits, labels)


def get_mc_task_dataloader(
        dataset_uri: str,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_seq_len: int,
        eos_tok_id: int,
        num_fewshot: int,
        preamble_string: str,  # e.g. 'translate english to french:'
        example_delimiter: str,  # e.g. '\n'
        continuation_delimiter: str,  # e.g. ''
) -> DataSpec:
    dataset = InContextLearningMultipleChoiceTaskDataset(dataset_uri, tokenizer, max_seq_len, eos_tok_id, num_fewshot,
                                             preamble_string, example_delimiter, continuation_delimiter)
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=True)
    batch_size = max(dataset.num_choices, batch_size)
    effective_batchsize = batch_size // dataset.num_choices
    print(f'Using microbatch size {effective_batchsize}')
    return DataSpec(DataLoader(
        dataset,
        batch_size=effective_batchsize,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
    ),
                    device_transforms=None,
                    get_num_samples_in_batch=dataset.get_num_samples_in_batch)
