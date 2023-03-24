# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from composer.core import DataSpec, Evaluator
from composer.utils import MissingConditionalImportError, dist, get_file

if TYPE_CHECKING:
    import transformers

__all__ = [
    'InContextLearningLMTaskDataset', 'InContextLearningMultipleChoiceTaskDataset', 'get_icl_task_dataloader',
    'make_evaluators', 'get_dataloaders_with_category'
]


def _make_padded_input(context_enc, continuation_enc, max_seq_len, pad_tok_id, padding_side='right'):
    if len(continuation_enc) + len(context_enc) > max_seq_len:
        # clip from the end
        context_max_subseq_len = max_seq_len - len(continuation_enc)

        if context_max_subseq_len < 0:
            raise Exception(f'Dataset included continuation longer than the max seq len')
            # can't support continuations which are longer than the max seq len

        context_enc = context_enc[-(context_max_subseq_len):]

    # continuation span is the _inclusive_ range of indices corresponding to the continuation
    continuation_span = torch.tensor(range(len(context_enc), len(context_enc) + len(continuation_enc)))
    inp = torch.tensor(
        (context_enc + continuation_enc),
        dtype=torch.long,
    )
    (inp_len,) = inp.shape

    # pad length from seq to padding_length
    if padding_side == 'right':
        inp = torch.cat(
            [
                inp,  # [seq]
                torch.LongTensor((max_seq_len - inp_len) * [pad_tok_id]),
            ],
            dim=0,
        )
    elif padding_side == 'left':
        inp = torch.cat(
            [
                torch.LongTensor((max_seq_len - inp_len) * [pad_tok_id]),
                inp,  # [seq]
            ],
            dim=0,
        )
    else:
        raise ValueError(f"Unknown padding_side {padding_side}. padding_side must be either 'left' or 'right'")

    return inp, continuation_span


def _get_fewshot_sample_idxs(dataset_size, num_fewshot, sample_idx):
    # samples without replacement. if num_fewshot exceeds the number of unique samples,
    # then we will have fewer than num_fewshot examples in context
    num_fewshot = min(dataset_size - 1, num_fewshot)
    fewshot_idxs = set(random.sample(range(0, dataset_size), num_fewshot))

    if sample_idx in fewshot_idxs:
        fewshot_idxs.remove(sample_idx)
        if len(fewshot_idxs) >= dataset_size - 1:
            return fewshot_idxs

        replacement_sample = random.choice(range(0, dataset_size))
        while replacement_sample in fewshot_idxs or replacement_sample == sample_idx:
            replacement_sample = random.choice(range(0, dataset_size))
        fewshot_idxs.add(replacement_sample)
    return fewshot_idxs


def _get_viable_candidates(entity, sample_index, samples, has_entities):
    viable_candidates = []
    if has_entities:
        if entity is None:
            for e in samples:
                for idx, sample in enumerate(samples[e]):
                    if e == entity and idx == sample_index:
                        continue
                    viable_candidates.append(sample)

            viable_candidates = [
                sample for e in samples for idx, sample in enumerate(samples[e]) if idx != sample_index
            ]
        else:
            for e in samples:
                if e == entity:
                    continue
                for _, sample in enumerate(samples[e]):
                    viable_candidates.append(sample)
    else:
        viable_candidates = [sample for idx, sample in enumerate(samples) if idx != sample_index]

    return viable_candidates


def _construct_fewshot_context(viable_candidates, num_fewshot, context_key, continuation_key, prompt_string,
                               continuation_delimiter, example_delimiter, question_prelimiter):

    preamble = prompt_string

    if num_fewshot > 0:
        fewshot_idxs = _get_fewshot_sample_idxs(len(viable_candidates), num_fewshot, -1)
        for fewshot_idx in fewshot_idxs:
            ctxt, cont = viable_candidates[fewshot_idx][context_key], viable_candidates[fewshot_idx][continuation_key]
            ctxt = f'{question_prelimiter}{ctxt}'
            if len(preamble) > 0:
                ctxt = f'{example_delimiter}{ctxt}'
            preamble += f'{ctxt}{continuation_delimiter}{cont}'

    return preamble


def _construct_multiple_choice_fewshot_context(
    viable_candidates,
    num_fewshot,
    context_key,
    continuation_key,
    choice_idx_key,
    prompt_string,
    continuation_delimiter,
    example_delimiter,
    question_prelimiter,
):

    preamble = prompt_string

    if num_fewshot > 0:
        fewshot_idxs = _get_fewshot_sample_idxs(len(viable_candidates), num_fewshot, -1)
        for fewshot_idx in fewshot_idxs:
            query, choices, gold_idx = viable_candidates[fewshot_idx][context_key], viable_candidates[fewshot_idx][
                continuation_key], viable_candidates[fewshot_idx][choice_idx_key]
            ctxt = f'{question_prelimiter}{query}'
            if len(preamble) > 0:
                ctxt = f'{example_delimiter}{ctxt}'
            preamble += f'{query}{continuation_delimiter}{choices[gold_idx]}'

    return preamble


def _encode_multiple_choice_example(entry, preamble, tokenizer, context_key, continuation_key, choice_idx_key,
                                    example_delimiter, continuation_delimiter, tokenize_continuation,
                                    prepend_space_to_continuation):

    encoded_example = {}
    query, choices, gold_idx = entry[context_key], entry[continuation_key], entry[choice_idx_key],

    if len(preamble) > 0:
        query = f'{example_delimiter}{query}'

    # rstrip the continuation delimiter, because the prompt ending in a space results in degenerate output
    continuation_delimiter_stripped = continuation_delimiter.rstrip()
    query = f'{query}{continuation_delimiter_stripped}'

    # tokenizers expect a space before new words. add a space to the continuation if there is none
    choices = [('' if c.startswith(' ') or not prepend_space_to_continuation else ' ') + c for c in choices]

    encoded_example['preamble'] = tokenizer(
        preamble
    )  # if the preamble is empty then these will be 0-length lists, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
    encoded_example[choice_idx_key] = gold_idx
    encoded_example[context_key] = tokenizer(query, add_special_tokens=False)
    encoded_example[continuation_key] = [
        (tokenizer(choice, add_special_tokens=False) if tokenize_continuation else choice) for choice in choices
    ]

    return encoded_example


def _encode_example(entry, preamble, tokenizer, context_key, continuation_key, example_delimiter,
                    continuation_delimiter, question_prelimiter, tokenize_continuation, prepend_space_to_continuation):
    encoded_example = {}
    ctxt = entry[context_key]
    ctxt = f'{question_prelimiter}{ctxt}'
    if len(preamble) > 0:
        ctxt = f'{example_delimiter}{ctxt}'

    # rstrip the continuation delimiter, because the prompt ending in a space results in degenerate output
    continuation_delimiter_stripped = continuation_delimiter.rstrip()
    ctxt = f'{ctxt}{continuation_delimiter_stripped}'

    # tokenizers expect a space before new words. add a space to the continuation if there is none
    if isinstance(entry[continuation_key], list):
        continuation = [
            ('' if c.startswith(' ') or not prepend_space_to_continuation else ' ') + c for c in entry[continuation_key]
        ]
    else:
        continuation = ('' if entry[continuation_key].startswith(' ') or not prepend_space_to_continuation else
                        ' ') + entry[continuation_key]

    # If the preamble is empty then this will be a 0-length list, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
    encoded_example['preamble'] = tokenizer(preamble)

    encoded_example[context_key] = tokenizer(ctxt, add_special_tokens=False)
    encoded_example[continuation_key] = tokenizer(continuation,
                                                  add_special_tokens=False) if tokenize_continuation else continuation

    return encoded_example


class InContextLearningQATaskDataset(Dataset):
    """A dataset that construct batches for in-context learning question answering evaluation

    The input format is expected to be a jsonl file with the following fields:
    - context: the question
    - answer: the preferred answer to the question
    - aliases: a list of aliases for the answer

    Args:
        dataset_uri (str): Either a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`. Dataset must consist of rows of JSON data points with "context",
            "answer", and "aliases". See tests/datasets/local_data/triviaqa_small.jsonl.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used to map between strings and token ids
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The maximum sequence length supported by the model
        pad_tok_id (int): The special token reserved for padding batches
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual (context, answer) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and answer in each example (e.g. '\nA: ')
        destination_path (str): Temporary path to store downloaded datasets
        question_prelimiter (str): String to put before each question (e.g. 'Q: ')
        padding_side (str): Whether to pad on the left or right side of the sequence
    """

    def __init__(self, dataset_uri: str, tokenizer: Union[transformers.PreTrainedTokenizer,
                                                          transformers.PreTrainedTokenizerFast], max_seq_len: int,
                 pad_tok_id: int, num_fewshot: int, prompt_string: str, example_delimiter: str,
                 continuation_delimiter: str, destination_path: str, question_prelimiter: str, padding_side: str,
                 category: Optional[str]):
        try:
            from datasets import load_dataset  # pyright: ignore [reportGeneralTypeIssues]
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='datasets',
                                                conda_channel='conda-forge') from e
        with dist.local_rank_zero_download_and_wait(destination_path):
            if dist.get_local_rank() == 0:
                get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        if category is not None and len(category) > 0:
            if 'category' not in next(iter(dataset)):
                raise Exception('Attempted to select sub-category of dataset with no category information.')
            if 'entity' in next(iter(dataset)):
                self.samples = {}
                for entry in dataset:
                    if entry['category'] != category:
                        continue
                    entity = entry.get('entity', None)

                    if entity not in self.samples:
                        self.samples[entity] = []

                    self.samples[entity].append({
                        'context': entry['context'],
                        'answer': entry['answer'],
                        'aliases': entry['aliases'],
                        'entity': entry.get('entity', None),
                    })
            else:
                self.samples = []
                for entry in dataset:
                    if entry['category'] != category:
                        continue
                    entity = entry.get('entity', None)
                    self.samples.append({
                        'context': entry['context'],
                        'answer': entry['answer'],
                        'aliases': entry['aliases'],
                        'entity': entry.get('entity', None),
                    })
        else:
            self.samples = list(
                dataset.map(lambda examples: {
                    'context': examples['context'],
                    'answer': examples['answer'],
                    'aliases': examples['aliases']
                }))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.padding_side = padding_side
        self.max_answer_length = 0
        if category is not None and isinstance(self.samples, dict):
            self.encoded_dataset = self.prep_examples_with_entities(num_fewshot, prompt_string, example_delimiter,
                                                                    continuation_delimiter, question_prelimiter)
        else:
            self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter,
                                                      continuation_delimiter, question_prelimiter)

    def prep_examples_with_entities(self, num_fewshot: int, prompt_string: str, example_delimiter: str,
                                    continuation_delimiter: str, question_prelimiter: str):
        max_answer_length = 0
        examples = []
        for entity in self.samples:
            for sample_index, entry in enumerate(self.samples[entity]):
                viable_prompt_candidates = _get_viable_candidates(entity, sample_index, self.samples, True)
                preamble = _construct_fewshot_context(viable_prompt_candidates, num_fewshot, 'context', 'answer',
                                                      prompt_string, continuation_delimiter, example_delimiter,
                                                      question_prelimiter)
                encoded_example = _encode_example(entry, preamble, self.tokenizer, 'context', 'aliases',
                                                  example_delimiter, continuation_delimiter, question_prelimiter, False,
                                                  False)

                examples.append(encoded_example)

                max_answer_length = max(max_answer_length,
                                        max(map(lambda x: len(self.tokenizer(x)['input_ids']), entry['aliases'])))

        self.max_answer_length = max_answer_length
        return examples

    def prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, continuation_delimiter: str,
                      question_prelimiter: str):
        """Prepares a set of language modeling tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            continuation_delimiter (str): The delimiter used to separate each context from its continuation
            question_prelimiter (str): The text to prepend to each question

        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        max_answer_length = 0
        examples = []
        for sample_index in tqdm(range(len(self.samples))):
            viable_prompt_candidates = _get_viable_candidates(None, sample_index, self.samples, False)
            preamble = _construct_fewshot_context(viable_prompt_candidates, num_fewshot, 'context', 'answer',
                                                  prompt_string, continuation_delimiter, example_delimiter,
                                                  question_prelimiter)
            encoded_example = _encode_example(self.samples[sample_index], preamble, self.tokenizer, 'context',
                                              'aliases', example_delimiter, continuation_delimiter, question_prelimiter,
                                              False, False)
            examples.append(encoded_example)

            max_answer_length = max(
                max_answer_length,
                max(map(lambda x: len(self.tokenizer(x)['input_ids']), self.samples[sample_index]['aliases'])))

        self.max_answer_length = max_answer_length
        return examples

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

    def collate_fn(self, data):
        inputs, answers = [], []
        for sample in data:
            preamble, context, aliases = (sample['preamble'], sample['context'], sample['aliases'])
            context_enc = preamble['input_ids'] + context['input_ids']
            inp, _ = _make_padded_input(context_enc, [],
                                        self.max_seq_len - self.max_answer_length,
                                        self.pad_tok_id,
                                        padding_side=self.padding_side)

            inputs.append(inp)
            answers.append(aliases)

        batch = {
            'input_ids': torch.stack(inputs),
            'mode': 'generate',
            'labels': answers,
            'generation_length': self.max_answer_length,
            'generation_kwargs': {
                'pad_token_id': self.pad_tok_id
            }
        }

        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0]


class InContextLearningLMTaskDataset(Dataset):
    """A dataset that construct batches for in-context learning language modeling evaluation

    Args:
        dataset_uri (str): Either a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`. Dataset must consist of rows of JSON data points with "context",
            and "continuation". See tests/datasets/local_data/lambada_small.jsonl.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used to transform data into batches
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The sequence length expected by the model
        pad_tok_id (int): The special token reserved for padding the ends of batches
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
        category (Optional[str])
    """

    def __init__(
        self,
        dataset_uri: str,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        max_seq_len: int,
        pad_tok_id: int,
        num_fewshot: int,
        prompt_string: str,
        example_delimiter: str,
        continuation_delimiter: str,
        destination_path: str,
        category: Optional[str],
    ):
        try:
            from datasets import load_dataset  # pyright: ignore [reportGeneralTypeIssues]
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='datasets',
                                                conda_channel='conda-forge') from e
        with dist.local_rank_zero_download_and_wait(destination_path):
            if dist.get_local_rank() == 0:
                get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)

        if category is not None and len(category) > 0:
            if 'category' not in next(iter(dataset)):
                raise Exception('Attempted to select sub-category of dataset with no category information.')
            if 'entity' in next(iter(dataset)):
                self.samples = {}
                for entry in dataset:
                    if entry['category'] != category:
                        continue
                    entity = entry.get('entity', None)

                    if entity not in self.samples:
                        self.samples[entity] = []

                    self.samples[entity].append({
                        'continuation': entry['continuation'],
                        'context': entry['context'],
                        'entity': entry.get('entity', None),
                    })
            else:
                self.samples = []
                for entry in dataset:
                    if entry['category'] != category:
                        continue
                    entity = entry.get('entity', None)
                    self.samples.append({
                        'continuation': entry['continuation'],
                        'context': entry['context'],
                        'entity': entry.get('entity', None),
                    })
        else:
            self.samples = list(
                dataset.map(lambda examples: {
                    'context': examples['context'],
                    'continuation': examples['continuation'],
                }))

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        if category is not None and isinstance(self.samples, dict):
            self.encoded_dataset = self.prep_examples_with_entities(num_fewshot, prompt_string, example_delimiter,
                                                                    continuation_delimiter)
        else:
            self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter,
                                                      continuation_delimiter)

    def prep_examples_with_entities(self, num_fewshot: int, prompt_string: str, example_delimiter: str,
                                    continuation_delimiter: str):
        """Prepares a set of language modeling tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            continuation_delimiter (str): The delimiter used to separate each context from its continuation

        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        examples = []

        for entity in self.samples:
            for sample_idx, entry in enumerate(self.samples[entity]):
                viable_prompt_candidates = _get_viable_candidates(entity, sample_idx, self.samples, True)
                preamble = _construct_fewshot_context(viable_prompt_candidates, num_fewshot, 'context', 'continuation',
                                                      prompt_string, continuation_delimiter, example_delimiter, '')
                encoded_example = _encode_example(entry, preamble, self.tokenizer, 'context', 'continuation',
                                                  example_delimiter, continuation_delimiter, '', True, True)
                examples.append(encoded_example)

        return examples

    def prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, continuation_delimiter: str):
        """Prepares a set of language modeling tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            continuation_delimiter (str): The delimiter used to separate each context from its continuation

        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):
            viable_prompt_candidates = _get_viable_candidates(None, sample_idx, self.samples, False)
            preamble = _construct_fewshot_context(viable_prompt_candidates, num_fewshot, 'context', 'continuation',
                                                  prompt_string, continuation_delimiter, example_delimiter, '')
            encoded_example = _encode_example(self.samples[sample_idx], preamble, self.tokenizer, 'context',
                                              'continuation', example_delimiter, continuation_delimiter, '', True, True)
            examples.append(encoded_example)

        return examples

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

    def collate_fn(self, data):
        inputs = []
        continuation_indices = []
        for data_pair in data:
            preamble, context, continuation = (data_pair['preamble'], data_pair['context'], data_pair['continuation'])

            context_enc = preamble['input_ids'] + context['input_ids']
            continuation_enc = continuation['input_ids']

            inp, continuation_span = _make_padded_input(context_enc, continuation_enc, self.max_seq_len,
                                                        self.pad_tok_id)

            inputs.append(inp)
            continuation_indices.append(continuation_span)

        batch = {
            'input_ids': torch.stack(inputs),
            'continuation_indices': continuation_indices,
            'mode': 'icl_task',
            'labels': torch.stack(inputs),
        }

        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0]


class InContextLearningMultipleChoiceTaskDataset(Dataset):
    """A dataset that construct batches for in-context learning multiple choice evaluation

    If each question has N answer choices, we construct N distinct inputs per question. In order to ensure
    consistency across multi-GPU, we set the batch size to be `min(N, batch_size)` so that all N
    inputs per question can stored in the same batch.

    Each batch then consists of batch_size // N distinct questions and has the following the structure

    'input_ids': Input tensor batch x seqlen x # tokens
    'continuation_indices': List of |batch| consisting of tensors indicating which indices in the sequence correspond to the question answer (aka continuation)
    'mode': Indicates to the model that this is an ICL task and may rely on a custom code path to properly update metrics
    'labels': Identical to the input, used by the model to calculate loss/metrics
    'gold_indices': List of length |batch_size // N| indicating for each question, which of the answers is correct (via an integer [0, N-1])
    'choice_groupings': Indicates which indices of the batch correspond to which questions

    Args:
        dataset_uri (str): Either a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`. Dataset must consist of rows of JSON data points with "query",
            "choices", and "gold" index. See tests/datasets/local_data/piqa_small.jsonl.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used to transform data into batches
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The sequence length expected by the model
        pad_tok_id (int): The special token reserved for padding the ends of batches
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
    """

    def __init__(self, dataset_uri: str, tokenizer: Union[transformers.PreTrainedTokenizer,
                                                          transformers.PreTrainedTokenizerFast], max_seq_len: int,
                 pad_tok_id: int, num_fewshot: int, prompt_string: str, example_delimiter: str,
                 continuation_delimiter: str, destination_path: str, category: Optional[str]):
        try:
            from datasets import load_dataset  # pyright: ignore [reportGeneralTypeIssues]
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='datasets',
                                                conda_channel='conda-forge') from e

        with dist.local_rank_zero_download_and_wait(destination_path):
            if dist.get_local_rank() == 0:
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
        self.pad_tok_id = pad_tok_id
        self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter, continuation_delimiter)

        if category is not None and len(category) > 0:
            if 'category' not in next(iter(dataset)):
                raise Exception('Attempted to select sub-category of dataset with no category information.')
            if 'entity' in next(iter(dataset)):
                self.samples = {}
                for entry in dataset:
                    if entry['category'] != category:
                        continue
                    entity = entry.get('entity', None)

                    if entity not in self.samples:
                        self.samples[entity] = []

                    self.samples[entity].append({
                        'query': entry['query'],
                        'choices': entry['choices'],
                        'gold': entry['gold'],
                        'entity': entry.get('entity', None),
                    })
            else:
                self.samples = []
                for entry in dataset:
                    if entry['category'] != category:
                        continue
                    entity = entry.get('entity', None)
                    self.samples.append({
                        'query': entry['query'],
                        'choices': entry['choices'],
                        'gold': entry['gold'],
                        'entity': entry.get('entity', None),
                    })
        else:
            self.samples = list(
                dataset.map(lambda entry: {
                    'query': entry['query'],
                    'choices': entry['choices'],
                    'gold': entry['gold'],
                }))

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        if category is not None and isinstance(self.samples, dict):
            self.encoded_dataset = self.prep_examples_with_entities(num_fewshot, prompt_string, example_delimiter,
                                                                    continuation_delimiter)
        else:
            self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter,
                                                      continuation_delimiter)

    def prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, continuation_delimiter: str):
        """Prepares a set of multiple choice questions into tokenized format with prompt and few shot examples.

        Each question consists of a query and set of answer choices, only one of which is correct. At inference time
        we construct individual inference examples consisting of the query + a single choice, as well as an optional (prompt) and optional list
        of example query + correct answers, which precede the test query + choice.

        For multiple choice, this method provides information relaying which of the answer choices is the correct one. This
        information is used for computing accuracy metrics.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each example query/answer pair
            continuation_delimiter (str): The delimiter used to separate each query from its answer

        Returns:
            dict: Contains the query, the list of encoded potential answer choices, the preamble (prompt + fewshot examples), and
                the index of the correct answer choice.
        """
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):

            viable_prompt_candidates = _get_viable_candidates(None, sample_idx, self.samples, False)
            preamble = _construct_multiple_choice_fewshot_context(viable_prompt_candidates, num_fewshot, 'query',
                                                                  'choices', 'gold', prompt_string,
                                                                  continuation_delimiter, example_delimiter, '')
            encoded_example = _encode_multiple_choice_example(self.samples[sample_idx], preamble, self.tokenizer,
                                                              'query', 'choices', 'gold', example_delimiter,
                                                              continuation_delimiter, True, True)
            examples.append(encoded_example)

        return examples

    def prep_examples_with_entities(self, num_fewshot: int, prompt_string: str, example_delimiter: str,
                                    continuation_delimiter: str):
        """Prepares a set of language modeling tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            continuation_delimiter (str): The delimiter used to separate each context from its continuation

        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        examples = []

        for entity in self.samples:
            for sample_idx, entry in enumerate(self.samples[entity]):

                viable_prompt_candidates = _get_viable_candidates(entity, sample_idx, self.samples, True)
                preamble = _construct_multiple_choice_fewshot_context(viable_prompt_candidates, num_fewshot, 'query',
                                                                      'choices', 'gold', prompt_string,
                                                                      continuation_delimiter, example_delimiter, '')
                encoded_example = _encode_multiple_choice_example(entry, preamble, self.tokenizer, 'query', 'choices',
                                                                  'gold', example_delimiter, continuation_delimiter,
                                                                  True, True)
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
            preamble, context, choices, gold_idx = (data_pair['preamble'], data_pair['query'], data_pair['choices'],
                                                    data_pair['gold'])

            for choice in choices:
                context_enc = preamble['input_ids'] + context['input_ids']
                continuation_enc = choice['input_ids']
                inp, continuation_span = _make_padded_input(context_enc, continuation_enc, self.max_seq_len,
                                                            self.pad_tok_id)

                inputs.append(inp)
                continuation_indices.append(continuation_span)

            gold_idxs.append(gold_idx)
            choice_end_idx = len(continuation_indices)
            choice_groupings.append((choice_start_idx, choice_end_idx))

        # We run each distinct query + answer choice through the model separately and determine which
        # answer has the lowest per-token-perplexity.
        #
        # If each question has N possible choices, all N must be grouped together as distinct elements of the batch
        # since the batch may consist of multiple questions, the choice_groupings indicates
        # which contiguous sequences of elements in the batch correspond to which question
        # gold_indices indicates which of the [0, N-1] choices is the correct one for each question.
        batch = {
            'input_ids': torch.stack(inputs),
            'continuation_indices': continuation_indices,
            'mode': 'icl_task',
            'labels': torch.stack(inputs),
            'gold_indices': gold_idxs,
            'choice_groupings': choice_groupings
        }
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0]

    def split_batch(self, batch: Any, microbatch_size: int):
        if self.get_num_samples_in_batch(batch) // self.num_choices > microbatch_size:
            raise Exception('Multiple choice tasks do not currently support batch splitting. Please set '
                            'dataloader batch size to a value less than or equal to the microbatch size. '
                            'Accordingly, auto microbatching does not work, so the microbatch size '
                            'should be manually set if using a batch size which does not fit in memory.')
        return [batch]


def get_icl_task_dataloader(
        icl_task_type: str,
        dataset_uri: str,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        batch_size: int,
        max_seq_len: int,
        pad_tok_id: int,
        num_fewshot: int,
        prompt_string: str,  # e.g. 'translate english to french:'
        example_delimiter: str,  # e.g. '\n'
        continuation_delimiter: str,  # e.g. ''
        destination_path: str,
        question_prelimiter: str = '',  # e.g. 'Question: '
        padding_side: str = 'left',
        category: Optional[str] = None):
    """This constructs a dataloader capable of evaluating LLMs on in-context learning language modeling tasks, for example LAMBADA. An example usage is below:
    >>> dl = get_icl_task_dataloader(
       ... 'language_modeling',
       ... dataset_uri,
       ... tokenizer,
       ... batch_size=2,
       ... max_seq_len=2048,
       ... pad_tok_id=tokenizer.pad_token_id,
       ... num_fewshot=10,
       ... prompt_string='translate english to french',
       ... example_delimiter='\n',
       ... continuation_delimiter=''
       )
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
        pad_tok_id (int): The special token reserved for padding the ends of batches
        num_fewshot (int): The number of complete fewshot examples to pad each test example with
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual examples (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
    Returns:
        DataLoader: A dataloader used for performing in-context learning evaluation on the dataset provided.
    """
    if icl_task_type == 'multiple_choice':
        dataset = InContextLearningMultipleChoiceTaskDataset(dataset_uri,
                                                             tokenizer,
                                                             max_seq_len,
                                                             pad_tok_id,
                                                             num_fewshot,
                                                             prompt_string,
                                                             example_delimiter,
                                                             continuation_delimiter,
                                                             destination_path=destination_path,
                                                             category=category)
        batch_size = max(dataset.num_choices, batch_size)
        effective_batchsize = batch_size // dataset.num_choices
    elif icl_task_type == 'language_modeling':
        dataset = InContextLearningLMTaskDataset(dataset_uri,
                                                 tokenizer,
                                                 max_seq_len,
                                                 pad_tok_id,
                                                 num_fewshot,
                                                 prompt_string,
                                                 example_delimiter,
                                                 continuation_delimiter,
                                                 destination_path=destination_path,
                                                 category=category)
        effective_batchsize = batch_size
    elif icl_task_type == 'question_answering':
        dataset = InContextLearningQATaskDataset(dataset_uri,
                                                 tokenizer,
                                                 max_seq_len,
                                                 pad_tok_id,
                                                 num_fewshot,
                                                 prompt_string,
                                                 example_delimiter,
                                                 continuation_delimiter,
                                                 destination_path=destination_path,
                                                 question_prelimiter=question_prelimiter,
                                                 padding_side=padding_side,
                                                 category=category)
        effective_batchsize = batch_size
    else:
        raise Exception(f'Unrecognized ICL task type: {icl_task_type}')

    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=effective_batchsize,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        ),
        device_transforms=None,
        get_num_samples_in_batch=dataset.get_num_samples_in_batch,
        split_batch=dataset.split_batch if isinstance(dataset, InContextLearningMultipleChoiceTaskDataset) else None)


def get_dataloaders_with_category(
    icl_task_type: str,
    dataset_uri: str,
    categories: List[str],
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    batch_size: int,
    max_seq_len: int,
    pad_tok_id: int,
    num_fewshot: int,
    prompt_string: str,  # e.g. 'translate english to french:'
    example_delimiter: str,  # e.g. '\n'
    continuation_delimiter: str,  # e.g. ''
    destination_path: str,
    question_prelimiter: str = '',  # e.g. 'Question: '
    padding_side: str = 'left',
) -> Dict[str, DataSpec]:
    """This constructs a dataloader capable of evaluating LLMs on in-context learning language modeling tasks, for example LAMBADA. An example usage is below:

    >>> dl = get_icl_task_dataloader(
       ... 'language_modeling',
       ... dataset_uri,
       ... tokenizer,
       ... batch_size=2,
       ... max_seq_len=2048,
       ... pad_tok_id=tokenizer.pad_token_id,
       ... num_fewshot=10,
       ... prompt_string='translate english to french',
       ... example_delimiter='\n',
       ... continuation_delimiter=''
       )
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
        pad_tok_id (int): The special token reserved for padding the ends of batches
        num_fewshot (int): The number of complete fewshot examples to pad each test example with
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual examples (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')

    Returns:
        DataLoader: A dataloader used for performing in-context learning evaluation on the dataset provided.
    """
    data_specs = {}
    for category in categories:
        dest = '/'.join(destination_path.split('/')[:-1]) + '/' + category + '_' + destination_path.split('/')[-1]
        data_specs[f'/{category}'] = get_icl_task_dataloader(
            icl_task_type,
            dataset_uri,
            tokenizer,
            batch_size,
            max_seq_len,
            pad_tok_id,
            num_fewshot,
            prompt_string,  # e.g. 'translate english to french:'
            example_delimiter,  # e.g. '\n'
            continuation_delimiter,  # e.g. ''
            dest,
            question_prelimiter,
            padding_side,
            category)

    data_specs[''] = get_icl_task_dataloader(
        icl_task_type,
        dataset_uri,
        tokenizer,
        batch_size,
        max_seq_len,
        pad_tok_id,
        num_fewshot,
        prompt_string,  # e.g. 'translate english to french:'
        example_delimiter,  # e.g. '\n'
        continuation_delimiter,  # e.g. ''
        destination_path,
        question_prelimiter,
        padding_side)
    return data_specs


def make_evaluators(
    base_label: str,
    metric_names: List[str],
    icl_task_type: str,
    dataset_uri: str,
    categories: List[str],
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    batch_size: int,
    max_seq_len: int,
    pad_tok_id: int,
    num_fewshot: int,
    prompt_string: str,  # e.g. 'translate english to french:'
    example_delimiter: str,  # e.g. '\n'
    continuation_delimiter: str,  # e.g. ''
    destination_path: str,
):

    dls = get_dataloaders_with_category(
        icl_task_type,
        dataset_uri,
        categories,
        tokenizer,
        batch_size,
        max_seq_len,
        pad_tok_id,
        num_fewshot,
        prompt_string,  # e.g. 'translate english to french:'
        example_delimiter,  # e.g. '\n'
        continuation_delimiter,  # e.g. ''
        destination_path,
    )

    return [Evaluator(label=base_label + k, dataloader=v, metric_names=metric_names) for k, v in dls.items()]
