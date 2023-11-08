# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from composer.core import DataSpec
from composer.core.data_spec import _default_split_batch, _split_list
from composer.utils import MissingConditionalImportError, dist, get_file

if TYPE_CHECKING:
    import transformers

# Allow models to have slightly more tokens than were used in the most verbose CoT in the dataset
_MAX_ANSWER_BUFFER_LENGTH = 10

__all__ = [
    'InContextLearningLMTaskDataset',
    'InContextLearningMultipleChoiceTaskDataset',
    'InContextLearningCodeEvalDataset',
    'InContextLearningQATaskDataset',
    'get_icl_task_dataloader',
]


def strip_data(samples):
    return [{k: v.strip() if isinstance(v, str) else v for k, v in entry.items()} for entry in samples]


def _tokenizer_needs_prefix_space(tokenizer) -> bool:
    # Test for whether a prefix space is needed before the continuation.
    # sentencepiece tokenization should not have a prefix space, but gpt2 style BPE should
    return len(tokenizer(' a', add_special_tokens=False)['input_ids']) == 1


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


def _get_fewshot_sample_idxs(dataset_size: int, num_fewshot: int, sample_idx: int, rng: random.Random):
    # samples without replacement. if num_fewshot exceeds the number of unique samples,
    # then we will have fewer than num_fewshot examples in context
    num_fewshot = min(dataset_size - 1, num_fewshot)
    fewshot_idxs = set(rng.sample(range(0, dataset_size), num_fewshot))

    if sample_idx in fewshot_idxs:
        fewshot_idxs.remove(sample_idx)
        if len(fewshot_idxs) >= dataset_size - 1:
            return fewshot_idxs

        replacement_sample = rng.choice(range(0, dataset_size))
        while replacement_sample in fewshot_idxs or replacement_sample == sample_idx:
            replacement_sample = rng.choice(range(0, dataset_size))
        fewshot_idxs.add(replacement_sample)
    return fewshot_idxs


class InContextLearningDataset(Dataset):

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
        fewshot_random_seed: int,
        icl_hf_loading_vars: dict = {},
        icl_hf_parsing_vars: dict = {},
        context_key: str = 'context',
        answer_key: str = 'answer',
        prelimiter: str = '',
        dont_split_keys: List[str] = [],
        normal_split_keys: List[str] = [],
        list_split_keys: List[str] = [],
    ):
        self.tokenizer = tokenizer
        self.prefix_space = _tokenizer_needs_prefix_space(self.tokenizer)

        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.padding_side = 'left'

        self.prelimiter = prelimiter
        self.example_delimiter = example_delimiter
        self.continuation_delimiter = continuation_delimiter
        self.context_key = context_key
        self.answer_key = answer_key

        self.samples = self._read_dataset(dataset_uri, destination_path, icl_hf_loading_vars, icl_hf_parsing_vars)
        self.samples = strip_data(self.samples)

        fewshot_rng = random.Random(fewshot_random_seed)
        self.encoded_dataset = self._prep_examples(num_fewshot, prompt_string, fewshot_rng)

        self.dont_split_keys = dont_split_keys
        self.normal_split_keys = normal_split_keys
        self.list_split_keys = list_split_keys

    def __getitem__(self, index: int):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

    def get_num_samples_in_batch(self, batch: dict) -> int:
        return batch['input_ids'].shape[0]

    def _read_dataset(self,
                      dataset_uri: str,
                      destination_path: str,
                      icl_hf_loading_vars: dict = None,
                      icl_hf_parsing_vars: dict = None):
        try:
            from datasets import load_dataset  # pyright: ignore [reportGeneralTypeIssues]
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='datasets',
                                                conda_channel='conda-forge') from e
        if 'hf://' in dataset_uri:
            dataset_uri = dataset_uri.replace('hf://', '')
            dataset = load_dataset(dataset_uri, split='train', **icl_hf_loading_vars)
        else:
            with dist.local_rank_zero_download_and_wait(destination_path):
                if dist.get_local_rank() == 0:
                    get_file(dataset_uri, destination_path, overwrite=True)
            dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        if icl_hf_parsing_vars:
            dataset = self._parse_hf_dataset(dataset, icl_hf_parsing_vars)
        else:
            dataset = self._parse_dataset(dataset)
        return dataset

    def _parse_hf_dataset(self, dataset, icl_hf_parsing_vars):
        dataset = dataset.map(
            lambda example: {k: ''.join([str(example[col]) for col in v]) for k, v in icl_hf_parsing_vars})
        return dataset

    def _parse_dataset(self, dataset: Dataset) -> List[Dict[str, str]]:
        return list(
            dataset.map(lambda examples: {
                self.context_key: examples['context'],
                self.answer_key: examples['answer'],
            }))

    def generate_few_shot_text(self, num_fewshot: int, sample_idx: int, preamble: str,
                               fewshot_rng: random.Random) -> str:
        """Formats the prompt fewshot examples for test sample `sample_idx`.

        Randomly select `num_fewshot` samples from the dataset (not including the sample at `sample_idx`) and format
        them each as follows `{example_delimiter}{prelimiter}{context}{continuation_delimiter}{chain_of_thought}{cot_delimiter}{answer}`.

        `chain_of_thought` will default to empty if not present in the dataset but `context` and `answer` must be present.

        Returns the formatted prompt_string + concatenated list of formatted few shot examples.
        """
        few_shot_text = preamble

        if num_fewshot > 0:
            fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx, fewshot_rng)
            for fewshot_idx in fewshot_idxs:
                ctxt = self.construct_context(self.samples[fewshot_idx], few_shot_text, add_answer=True)
                few_shot_text += ctxt

        return few_shot_text

    def construct_context(self, sample: dict, preceding_text: str = '', add_answer: bool = False):
        ctxt = sample[self.context_key]
        ctxt = f'{self.prelimiter}{ctxt}'
        if len(preceding_text) > 0:
            ctxt = f'{preceding_text}{self.example_delimiter}{ctxt}'
        ctxt = f'{ctxt}{self.continuation_delimiter}'
        if add_answer:
            ctxt = f'{ctxt}{self.get_answer_from_sample(sample)}'
        return ctxt

    def get_answer_from_sample(self, sample: dict):
        return sample[self.answer_key]

    def fix_eos_on_preamble(self, preamble: dict):
        # If the preamble is empty then this will be a 0-length list, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
        # If there is an EOS token added, we need to remove it so it is not in the middle of the prompt
        if (self.tokenizer.eos_token_id is not None and len(preamble['input_ids']) > 1 and
                preamble['input_ids'][-1] == self.tokenizer.eos_token_id):
            preamble['input_ids'] = preamble['input_ids'][:-1]
        return preamble

    def tokenize_example(self, prompt_and_fewshot: str, ctxt: str):
        tokenized_example = {}
        preamble = self.tokenizer(prompt_and_fewshot)
        preamble = self.fix_eos_on_preamble(preamble)
        tokenized_example['preamble'] = preamble
        # rstrip context because a prompt ending in a space results in degenerate output
        #TODO: use diff key for this?
        tokenized_example['context'] = self.tokenizer(ctxt.rstrip(), add_special_tokens=False)
        return tokenized_example

    def _prep_examples(self, num_fewshot: int, prompt_string: str, fewshot_rng: random.Random) -> List[Dict[str, Any]]:
        """Prepares a set of language modeling tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            continuation_delimiter (str): The delimiter used to separate each context from its continuation
            fewshot_rng (random.Random): Random number generator to use for fewshot sampling
            cot_delimiter (str): The delimiter used to separate the chain-of-thought (if present) from the final model response.


        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):
            prompt_and_fewshot = self.generate_few_shot_text(num_fewshot, sample_idx, prompt_string, fewshot_rng)
            ctxt = self.construct_context(self.samples[sample_idx], prompt_and_fewshot, add_answer=False)
            tokenized_example = self.tokenize_example(prompt_and_fewshot, ctxt)
            tokenized_example = self.additional_processing_for_example(tokenized_example, self.samples[sample_idx])
            examples.append(tokenized_example)
        return examples

    def additional_processing_for_example(self, tokenized_example: dict, sample: dict):
        return tokenized_example

    def collate_fn(self, data):
        pass

    def split_batch(self, batch: Any, microbatch_size: int):
        # Don't split kwargs that don't change
        # Normally split torch tensors
        # List split lists of strings
        # no_split = ['mode', 'generation_length', 'pass_at_k', 'generation_kwargs']
        # normal_split = ['input_ids', 'attention_mask']
        # list_split = [
        #     'labels', 'tests', 'canonical_solutions', 'entry_points', 'test_inputs', 'test_outputs', 'prompts',
        #     'languages'
        # ]
        chunked = {}
        for k, v in batch.items():
            if k in self.dont_split_keys:
                # Defer broadcasting until we know num_chunks
                pass
            elif k in self.list_split_keys:
                chunked[k] = _split_list(v, microbatch_size)
            elif k in self.normal_split_keys:
                chunked[k] = _default_split_batch(v, microbatch_size)
            else:
                raise ValueError(f'Unexpected key {k}')
        num_chunks = len(chunked['input_ids'])
        for k, v in batch.items():
            if isinstance(v, (int, float, str, bool, dict)):
                chunked[k] = [v] * num_chunks

        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


class InContextLearningQATaskDataset(InContextLearningDataset):
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
        prelimiter (str): String to put before each question (e.g. 'Q: ')
        fewshot_random_seed (int): Random seed to use for fewshot sampling
    """

    def __init__(self, cot_delimiter: str = '', *args, **kwargs):
        super().__init__(dont_split_keys=['mode', 'generation_length', 'generation_kwargs', 'cot_delimiter'],
                         normal_split_keys=['input_ids', 'attention_mask'],
                         list_split_keys=['labels'],
                         *args,
                         **kwargs)
        self.cot_delimiter = cot_delimiter
        self.max_answer_length = self.get_max_answer_length()

    def _parse_dataset(self, dataset: Dataset) -> List[Dict[str, str]]:
        return list(
            dataset.map(
                lambda examples: {
                    'context': examples['context'],
                    'answer': examples['answer'],
                    'aliases': set([examples['answer']] + examples.get('aliases', [])),
                    'chain_of_thought': examples.get('chain_of_thought', '')
                }))

    def get_answer_from_sample(self, sample):
        # If we add the answer, we need to also add COT
        chain_of_thought = sample.get('chain_of_thought', '')
        if len(chain_of_thought) == 0:
            cot_delimiter = ''
        else:
            cot_delimiter = self.cot_delimiter
        return f'{self.continuation_delimiter}{chain_of_thought}{cot_delimiter}{sample[self.answer_key]}'

    def additional_processing_for_example(self, tokenized_example: dict, sample: dict):
        tokenized_example['aliases'] = list(sample['aliases'])
        tokenized_example['cot_delimiter'] = self.cot_delimiter
        return tokenized_example

    def get_max_answer_length(self):
        max_answer_length = 0
        for sample in self.samples:
            for answer in sample['aliases']:
                # DRY says I should use get_answer_from_sample somehow here
                response = f"{sample['chain_of_thought']}{self.cot_delimiter}{answer}"
                max_answer_length = max(max_answer_length, len(self.tokenizer(response)['input_ids']))
        max_answer_length = max_answer_length + (_MAX_ANSWER_BUFFER_LENGTH if len(self.cot_delimiter) > 0 else 0)
        return max_answer_length

    def collate_fn(self, data):
        inputs, answers = [], []
        cot_delimiter = ''

        for sample in data:
            preamble, context, aliases = (sample['preamble'], sample['context'], sample['aliases'])
            context_enc = preamble['input_ids'] + context['input_ids']
            inp, _ = _make_padded_input(context_enc, [],
                                        self.max_seq_len - self.max_answer_length,
                                        self.pad_tok_id,
                                        padding_side=self.padding_side)

            inputs.append(inp)
            answers.append(aliases)

            # We will search for the answer within the portion of the model response
            # beginning with `cot_delimiter`
            cot_delimiter = sample['cot_delimiter']

        batch = {
            'input_ids': torch.stack(inputs),
            'mode': 'generate',
            'labels': answers,
            'cot_delimiter': cot_delimiter,
            'generation_length': self.max_answer_length,
            'generation_kwargs': {
                'pad_token_id': self.pad_tok_id,
                'use_cache': True
            }
        }

        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch


class InContextLearningLMTaskDataset(InContextLearningDataset):
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
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
        fewshot_random_seed (int): Random seed used to select fewshot examples
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_dataset(self, dataset: Dataset) -> List[Dict[str, str]]:
        return list(
            dataset.map(lambda examples: {
                'continuation': examples['continuation'],
                'context': examples['context'],
            }))

    def additional_processing_for_example(self, tokenized_example: dict, sample: dict):
        cont = sample['continuation']
        if self.prefix_space and not cont.startswith(' '):
            cont = f' {cont}'
        tokenized_example['continuation'] = self.tokenizer(cont, add_special_tokens=False)
        return tokenized_example

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


class InContextLearningMultipleChoiceTaskDataset(InContextLearningDataset):
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
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
        fewshot_random_seed (int): Random seed used to select fewshot examples
    """

    def __init__(self, *args, **kwargs):
        super().__init__(dont_split_keys=['mode'],
                         real_split_keys=['input_ids', 'labels', 'attention_mask'],
                         normal_split_key=['gold_indices'],
                         *args,
                         **kwargs)
        self.num_choices = len(self.samples[0]['choices'])
        self.context_key = 'query'
        # self.dont_split_keys = ['mode']
        # self.real_split_keys = ['input_ids', 'labels', 'attention_mask']
        # self.normal_split_keys = ['gold_indices']

    def _parse_dataset(self, dataset: Dataset) -> List[Dict[str, str]]:
        return list(
            dataset.map(lambda examples: {
                'query': examples['query'],
                'choices': examples['choices'],
                'gold': examples['gold']
            }))

    def get_answer_from_sample(self, sample: dict):
        choices = sample['choices']
        gold_idx = sample['gold']
        return choices[gold_idx]

    def additional_processing_for_example(self, tokenized_example: dict, sample: dict):
        choices = sample['choices']
        if self.prefix_space:
            choices = [(f' {choice}' if not choice.startswith(' ') else choice) for choice in choices]
        tokenized_example['choices'] = [self.tokenizer(choice, add_special_tokens=False) for choice in choices]
        tokenized_example['gold_idx'] = sample['gold']
        return tokenized_example

    def collate_fn(self, data):
        inputs = []
        continuation_indices = []
        gold_idxs = []
        choice_groupings = []
        for data_pair in data:

            choice_start_idx = len(continuation_indices)
            preamble, context, choices, gold_idx = (data_pair['preamble'], data_pair['query'], data_pair['choices'],
                                                    data_pair['gold_idx'])

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
        return batch['input_ids'].shape[0] // self.num_choices

    def split_batch(self, batch: Any, microbatch_size: int):
        """Split batch while ensuring all continuations are in the same microbatch.

        In ICL Multiple Choice, we duplicate each data point for each possible continuation.
        When splitting a batch, we have logical samples, which refer to one possible question,
        and real samples, which refers to one possible continuation. As sample count and
        microbatch_size are tracked in logical samples, we split logical attributes by
        microbatch_size and real attributes by microbatch_size * num_choices.
        """
        # There are extra split options in this func for multiple choice
        chunked = {}
        for k, v in batch.items():
            if k in self.dont_split_keys:
                # Defer broadcasting primitives until we know num_chunks
                pass
            elif k == 'continuation_indices':
                # List of list, so we have to directly call _split_list
                chunked[k] = _split_list(v, microbatch_size * self.num_choices)
            elif k == 'choice_groupings':
                # List of list, so we have to directly call _split_list
                chunked[k] = _split_list(v, microbatch_size)
            elif k in self.real_split_keys:
                chunked[k] = _default_split_batch(v, microbatch_size * self.num_choices)
            elif k in self.normal_split_keys:
                chunked[k] = _default_split_batch(v, microbatch_size)
            else:
                raise ValueError(f'Unexpected key {k}')
        num_chunks = len(chunked['input_ids'])
        # Broadcast primitives to all chunks
        for k, v in batch.items():
            if isinstance(v, (int, float, str, bool)):
                chunked[k] = [v] * num_chunks
        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


class InContextLearningSchemaTaskDataset(InContextLearningMultipleChoiceTaskDataset):
    """A dataset that constructs batches for in-context learning schema evaluation
    A schema task involves sentences with a fill-in-the-blank where the user needs to choose the correct word
    to fill in from a set of N options. We use the partial evaluation technique from https://arxiv.org/abs/1806.02847
    to determine the model's choice of fill-in word.
    Each batch then consists of batch_size // N distinct tasks and has the following the structure
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
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
        fewshot_random_seed (int): Random seed used to select fewshot examples
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_dataset(self, dataset: Dataset) -> List[Dict[str, str]]:
        return list(
            dataset.map(
                lambda examples: {
                    'context_options': examples['context_options'],
                    'continuation': examples['continuation'],
                    'gold': examples['gold']
                }))

    def construct_context(self, sample, preceding_text: str = '', add_answer: bool = False):
        context_options = sample['context_options']
        gold_idx = sample['gold']
        continuation = sample['continuation']
        assert isinstance(gold_idx, int)
        # TODO: fix this?
        if add_answer:
            context = context_options[gold_idx]
            if len(preceding_text) > 0:
                context = f'{self.example_delimiter}{context}'
            context = f'{context}{self.continuation_delimiter}{continuation}'
        else:
            context_options = sample['context_options']
            if len(preceding_text) > 0:
                context_options = [f'{self.example_delimiter}{c}{self.continuation_delimiter}' for c in context_options]

        return context

    def generate_few_shot_text(self, num_fewshot: int, sample_idx: int, preamble: str,
                               fewshot_rng: random.Random) -> str:
        preamble = preamble
        if num_fewshot > 0:
            fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx, fewshot_rng)
            for fewshot_idx in fewshot_idxs:
                context = self.construct_context(self.samples[fewshot_idx])
                preamble += context
        return preamble

    def _prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, continuation_delimiter: str,
                       fewshot_rng: random.Random):
        """Prepares a set of schema questions into tokenized format with prompt and few shot examples.
        Each question consists of a set of possible contexts followed by a continuation, only one of the contexts would logically permit the continuation.
        At inference time we construct individual inference examples consisting of a single context option + the continuation,
        as well as an optional (prompt) and optional list of example correct context option + continuations, which precede the test context option + continuation.
        For schema, this method provides information relaying which of the answer choices is the correct one. This
        information is used for computing accuracy metrics.
        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each example query/answer pair
            continuation_delimiter (str): The delimiter used to separate each query from its answer
            fewshot_rng (random.Random): Random number generator used to select fewshot examples
        Returns:
            dict: Contains the query, the list of encoded potential answer choices, the preamble (prompt + fewshot examples), and
                the index of the correct answer choice.
        """

        # TODO: fix this
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):
            prompt_and_fewshot = self.generate_few_shot_text(num_fewshot, sample_idx, prompt_string, fewshot_rng)
            tokenized_example = self.tokenize_example(prompt_and_fewshot, '')
            tokenized_example['context_options'] = [
                self.tokenizer(c, add_special_tokens=False) for c in context_options
            ]
            examples.append(tokenized_example)

        return examples

    def additional_processing_for_example(self, tokenized_example: dict, sample: dict):
        continuation = sample['continuation']
        if self.prefix_space:
            continuation = f' {continuation}' if not continuation.startswith(' ') else continuation
        tokenized_example['continuation'] = self.tokenizer(continuation, add_special_tokens=False)
        tokenized_example['gold_idx'] = sample['gold_idx']
        return tokenized_example

    def collate_fn(self, data):
        inputs = []
        continuation_indices = []
        gold_idxs = []
        choice_groupings = []
        for data_pair in data:

            continuation_start_idx = len(continuation_indices)
            preamble, context_options, continuation, gold_idx = (data_pair['preamble'], data_pair['context_options'],
                                                                 data_pair['continuation'], data_pair['gold_idx'])

            for ctxt in context_options:
                context_enc = preamble['input_ids'] + ctxt['input_ids']
                continuation_enc = continuation['input_ids']
                inp, continuation_span = _make_padded_input(context_enc, continuation_enc, self.max_seq_len,
                                                            self.pad_tok_id)

                inputs.append(inp)
                continuation_indices.append(continuation_span)

            gold_idxs.append(gold_idx)
            continuation_end_idx = len(continuation_indices)
            choice_groupings.append((continuation_start_idx, continuation_end_idx))

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


class InContextLearningCodeEvalDataset(InContextLearningDataset):
    """ A dataset that constructs batches for in-context learning code evaluation

    The input format is expected to be a jsonl file with the following fields:
    - task_id: label of given task
    - prompt: the code snippet that must be completed
    - entry_point: the entry to the function/code snippet to generate
    - canonical_solution: working solution
    - test: the checker code that will run to completion if the code generation is valid and otherwise throw assertion
    - test_inputs: list of test inputs
    - test_outputs: list of test outputs
    - language: the language of the code snippet
    Args:
        dataset_uri (str): Either a local path, or a remote path beginning with ``s3://``, or another backend
        supported by :meth:`composer.utils.maybe_create_object_store_from_uri`. Dataset must consist of rows of JSON data points with "task_id",
        "prompt", "entry_point", "canonical_solution", "test", "test_inputs", and "test_outputs". See tests/datasets/local_data/human_eval_small.jsonl.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used to map between strings and token ids
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The maximum sequence length supported by the model
        pad_tok_id (int): The special token reserved for padding batches
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual (context, answer) pairs (e.g. '\n')
        destination_path (str): Temporary path to store downloaded datasets
        code_prelimiter (str): String to put before each code prompt (e.g. 'Q: ')
        fewshot_random_seed (int): Random seed to use for fewshot sampling
        generations_per_sample: how many outputs to generate per prompt
        top_p: top_p sampling parameter for nucleus sampling
        top_k: top_k sampling parameter for number of samples to consider
    """

    def __init__(self,
                 generations_per_sample: int,
                 pass_at_k: int = 1,
                 top_p: Optional[float] = 0.95,
                 top_k: Optional[int] = 40,
                 *args,
                 **kwargs):
        if generations_per_sample < pass_at_k:
            raise ValueError(
                f'generations_per_sample ({generations_per_sample}) must be greater than or equal to pass_at_k ({pass_at_k}) for code evaluation.'
            )

        super().__init__(context_key='prompt', answer_key='canonical_solution', *args, **kwargs)
        self.pass_at_k = pass_at_k
        self.generations_per_sample = generations_per_sample
        self.max_prompt_length = self.get_max_prompt_length()
        self.top_p = top_p
        self.top_k = top_k

        self.dont_split_keys = ['mode', 'generation_length', 'pass_at_k', 'generation_kwargs']
        self.normal_split_keys = ['input_ids', 'attention_mask']
        self.list_split_keys = [
            'labels', 'tests', 'canonical_solutions', 'entry_points', 'test_inputs', 'test_outputs', 'prompts',
            'languages'
        ]

    def _parse_dataset(self, dataset: Dataset) -> List[Dict[str, str]]:
        return list(
            dataset.map(
                lambda examples: {
                    'task_id': examples['task_id'],
                    'prompt': examples['prompt'],
                    'canonical_solution': examples['canonical_solution'],
                    'test': examples['test'],
                    'entry_point': examples['entry_point'],
                    'test_inputs': examples['test_inputs'],
                    'test_outputs': examples['test_outputs'],
                    'language': examples['language'],
                }))

    def get_max_prompt_length(self):
        max_prompt_length = 0
        for sample in self.samples:
            max_prompt_length = max(max_prompt_length,
                                    len(sample['preamble']['input_ids'] + sample['prompt']['input_ids']))
        return max_prompt_length

    def additional_processing_for_example(self, tokenized_example: dict, sample: dict):
        tokenized_example['prompt_text'] = sample['prompt']
        tokenized_example['task_id'] = sample['task_id']
        tokenized_example['canonical_solution'] = sample['canonical_solution']
        tokenized_example['test'] = sample['test']
        tokenized_example['entry_point'] = sample['entry_point']
        tokenized_example['test_inputs'] = sample['test_inputs']
        tokenized_example['test_outputs'] = sample['test_outputs']
        tokenized_example['language'] = sample['language']
        return tokenized_example

    def collate_fn(self, data):
        inputs, prompts, tests, canonical_solutions, entry_points, test_inputs, test_outputs, languages = [], [], [], [], [], [], [], []
        for sample in data:
            preamble, prompt, text_prompt, canonical_solution, test, entry_point, test_input, test_output, language = (
                sample['preamble'],
                sample['prompt'],
                sample['prompt_text'],
                sample['canonical_solution'],
                sample['test'],
                sample['entry_point'],
                sample['test_inputs'],
                sample['test_outputs'],
                sample['language'],
            )
            context_enc = preamble['input_ids'] + prompt['input_ids']
            inp, _ = _make_padded_input(context_enc, [],
                                        self.max_prompt_length,
                                        self.pad_tok_id,
                                        padding_side=self.padding_side)

            inputs.append(inp)
            tests.append(test)
            prompts.append(text_prompt)
            canonical_solutions.append(canonical_solution)
            entry_points.append(entry_point)
            test_inputs.append(test_input)
            test_outputs.append(test_output)
            languages.append(language)

        batch = {
            'input_ids': torch.stack(inputs),
            'mode': 'generate',
            'labels': canonical_solutions,
            'prompts': prompts,  # list of prompts
            'tests': tests,  # list of tests
            'canonical_solutions': canonical_solutions,  # list of solutions
            'entry_points': entry_points,  # list of entry points
            'test_inputs': test_inputs,  # list of test inputs
            'test_outputs': test_outputs,  # list of test outputs
            'languages': languages,  # list of languages
            'pass_at_k': self.pass_at_k,
            'generation_length': self.max_seq_len - self.max_prompt_length,
            'generation_kwargs': {
                'pad_token_id': self.pad_tok_id,
                'num_beams': 1,  # single beam
                'num_return_sequences': self.generations_per_sample,  # how many gens per prompt
                'do_sample': True,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'use_cache': True,
            }
        }
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch


def build_icl_dataloader(
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
    prelimiter: str = '',  # e.g. 'Question: '
    cot_delimiter: str = '',
    fewshot_random_seed: int = 1234,
    pass_at_k: int = 1,
    generations_per_sample: int = 1,
) -> DataSpec:
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
                                                             fewshot_random_seed=fewshot_random_seed)
        batch_size = max(dataset.num_choices, batch_size)
        effective_batchsize = batch_size // dataset.num_choices
    elif icl_task_type == 'schema':
        dataset = InContextLearningSchemaTaskDataset(dataset_uri,
                                                     tokenizer,
                                                     max_seq_len,
                                                     pad_tok_id,
                                                     num_fewshot,
                                                     prompt_string,
                                                     example_delimiter,
                                                     continuation_delimiter,
                                                     destination_path=destination_path,
                                                     fewshot_random_seed=fewshot_random_seed)
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
                                                 fewshot_random_seed=fewshot_random_seed)
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
                                                 prelimiter=prelimiter,
                                                 fewshot_random_seed=fewshot_random_seed,
                                                 cot_delimiter=cot_delimiter)
        effective_batchsize = batch_size
    elif icl_task_type == 'code_evaluation':
        dataset = InContextLearningCodeEvalDataset(dataset_uri,
                                                   tokenizer,
                                                   max_seq_len,
                                                   pad_tok_id,
                                                   num_fewshot,
                                                   prompt_string,
                                                   example_delimiter,
                                                   destination_path=destination_path,
                                                   prelimiter=prelimiter,
                                                   fewshot_random_seed=fewshot_random_seed,
                                                   pass_at_k=pass_at_k,
                                                   generations_per_sample=generations_per_sample)
        effective_batchsize = batch_size
    else:
        raise Exception(f'Unrecognized ICL task type: {icl_task_type}')

    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)

    split_batch = None
    if isinstance(
            dataset,
        (InContextLearningMultipleChoiceTaskDataset, InContextLearningQATaskDataset, InContextLearningCodeEvalDataset)):
        split_batch = dataset.split_batch

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=effective_batchsize,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        ),
        device_transforms=None,
        get_num_samples_in_batch=dataset.get_num_samples_in_batch,
        split_batch=split_batch,
    )


def partition_dataset_by_category(dataset_uri: str, destination_path: str) -> Dict[str, str]:
    """If has_categories is enabled, we partition the dataset into a separate dataset for each category value in the data and write each partition to a local file.

    Args:
        dataset_uri (str): Location of dataset.
        destination_path (str): Base destination path, we will write a separate partition off this URI for each category.

    Raises:
        MissingConditionalImportError: If datasets not installed raise exception.
        Exception: If 'category' key missing from dataset, raise exception.
    Returns:
        Dict[str, str]: Mapping of category names to partitioned dataset local files names.
    """
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
    if 'category' not in dataset.features.keys():
        raise Exception(
            f"Attempted to partition dataset by `category` but it doesn't have a `category` key. Got keys: {str(list(dataset.features.keys()))}"
        )
    categories = sorted(set(dataset['category']))
    output_files = {}
    for cat in categories:
        path = destination_path.split('/')
        cat_dest = '/'.join(path[:-1]) + f'/{cat}_{path[-1]}'
        tmp_path_to_broadcast = str(os.path.abspath(cat_dest))
        gathered_paths = dist.all_gather_object(tmp_path_to_broadcast)
        if dist.get_local_rank() == 0:
            subset = [l for l in dataset if l['category'] == cat]
            with open(gathered_paths[0], 'w', encoding='utf8') as f:
                for l in subset:
                    f.write(json.dumps(l, ensure_ascii=False) + '\n')
        output_files[cat] = cat_dest
    return output_files


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
        continuation_delimiter: str = '',
        destination_path: str = '',
        prelimiter: str = '',  # e.g. 'Question: '
        fewshot_random_seed: int = 1234,
        pass_at_k: int = 1,
        generations_per_sample: int = 1,
        cot_delimiter: str = '',
        has_categories: bool = False) -> Union[DataSpec, Dict[str, DataSpec]]:
    """This constructs a dataloader (or dataloaders if has_categories is True) capable of evaluating LLMs on in-context learning language modeling tasks, for example LAMBADA. An example usage is below:

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
        destination_path: (str): This is the local file where remote datasets will be saved.
        prelimiter: (str): For QA tasks, this will be prepended to each question.
        has_categories: (bool): If ``True``, we will search the dataset file for a category key, and partition the dataset into a separate dataloader for each category occurring in the data.

    Returns:
        DataLoader: A dataloader used for performing in-context learning evaluation on the dataset provided.
    """

    if has_categories:
        result_dls = {}
        output_files = partition_dataset_by_category(dataset_uri, destination_path)
        categories = sorted(output_files.keys())
        for category in categories:
            partition_uri = output_files[category]
            result_dls[category] = build_icl_dataloader(
                icl_task_type,
                partition_uri,
                tokenizer,
                batch_size,
                max_seq_len,
                pad_tok_id,
                num_fewshot,
                prompt_string,
                example_delimiter,
                continuation_delimiter,
                partition_uri + '_tmp',
                prelimiter,
                cot_delimiter,
                fewshot_random_seed,
                pass_at_k,
                generations_per_sample,
            )
        return result_dls
    else:
        return build_icl_dataloader(
            icl_task_type,
            dataset_uri,
            tokenizer,
            batch_size,
            max_seq_len,
            pad_tok_id,
            num_fewshot,
            prompt_string,
            example_delimiter,
            continuation_delimiter,
            destination_path,
            prelimiter,
            cot_delimiter,
            fewshot_random_seed,
            pass_at_k,
            generations_per_sample,
        )
