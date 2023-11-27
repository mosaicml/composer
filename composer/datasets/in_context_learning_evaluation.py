# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
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
        fewshot_random_seed (int): Random seed to use for fewshot sampling
    """

    def _read_dataset(self, dataset: Dataset) -> List[Dict[str, str]]:
        result = []
        for example in dataset:
            result.append({
                'context': example['context'],
                'answer': example['answer'],
                'aliases': set([example['answer']] + example.get('aliases', [])),
                'chain_of_thought': example.get('chain_of_thought', '')
            })
        return result

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
        question_prelimiter: str,
        fewshot_random_seed: int,
        cot_delimiter: str = '',
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

        self.samples = self._read_dataset(dataset)
        self.samples = strip_data(self.samples)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.padding_side = 'left'
        self.max_answer_length = 0
        fewshot_rng = random.Random(fewshot_random_seed)
        self.encoded_dataset = self._prep_examples(num_fewshot, prompt_string, example_delimiter,
                                                   continuation_delimiter, question_prelimiter, fewshot_rng,
                                                   cot_delimiter)

    def _format_prompt_and_fewshot(self, num_fewshot: int, prompt_string: str, example_delimiter: str,
                                   continuation_delimiter: str, question_prelimiter: str, cot_delimiter: str,
                                   fewshot_rng: random.Random, sample_idx: int) -> str:
        """Formats the prompt fewshot examples for test sample `sample_idx`.

        Randomly select `num_fewshot` samples from the dataset (not including the sample at `sample_idx`) and format
        them each as follows `{example_delimiter}{question_prelimiter}{context}{continuation_delimiter}{chain_of_thought}{cot_delimiter}{answer}`.

        `chain_of_thought` will default to empty if not present in the dataset but `context` and `answer` must be present.

        Returns the formatted prompt_string + concatenated list of formatted few shot examples.
        """
        prompt_and_fewshot = prompt_string

        if num_fewshot > 0:
            fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx, fewshot_rng)
            for fewshot_idx in fewshot_idxs:
                context = self.samples[fewshot_idx]['context']
                chain_of_thought = self.samples[fewshot_idx].get('chain_of_thought', '')
                answer = self.samples[fewshot_idx]['answer']

                if len(chain_of_thought) == 0:
                    cot_delimiter = ''
                context = f'{question_prelimiter}{context}'
                if len(prompt_and_fewshot) > 0:
                    context = f'{example_delimiter}{context}'
                prompt_and_fewshot += f'{context}{continuation_delimiter}{chain_of_thought}{cot_delimiter}{answer}'

        return prompt_and_fewshot

    def _prep_examples(self,
                       num_fewshot: int,
                       prompt_string: str,
                       example_delimiter: str,
                       continuation_delimiter: str,
                       question_prelimiter: str,
                       fewshot_rng: random.Random,
                       cot_delimiter: str = '') -> List[Dict[str, Any]]:
        """Prepares a set of language modeling tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            continuation_delimiter (str): The delimiter used to separate each context from its continuation
            question_prelimiter (str): The text to prepend to each question
            cot_delimiter (str): The delimiter used to separate the chain-of-thought (if present) from the final model response.
            fewshot_rng (random.Random): Random number generator to use for fewshot sampling
            cot_delimiter (str): The delimiter used to separate the chain-of-thought (if present) from the final model response.


        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        max_answer_length = 0
        has_cot = False
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):
            encoded_example = {}

            prompt_and_fewshot = self._format_prompt_and_fewshot(num_fewshot, prompt_string, example_delimiter,
                                                                 continuation_delimiter, question_prelimiter,
                                                                 cot_delimiter, fewshot_rng, sample_idx)
            ctxt = self.samples[sample_idx]['context']
            ctxt = f'{question_prelimiter}{ctxt}'
            if len(prompt_and_fewshot) > 0:
                ctxt = f'{example_delimiter}{ctxt}'

            # rstrip the continuation delimiter, because the prompt ending in a space results in degenerate output
            continuation_delimiter_stripped = continuation_delimiter.rstrip()
            ctxt = f'{ctxt}{continuation_delimiter_stripped}'

            # If the preamble is empty then this will be a 0-length list, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
            encoded_example['preamble'] = self.tokenizer(prompt_and_fewshot)
            # If there is an EOS token added, we need to remove it so it is not in the middle of the prompt
            if self.tokenizer.eos_token_id is not None and len(
                    encoded_example['preamble']
                ['input_ids']) > 1 and encoded_example['preamble']['input_ids'][-1] == self.tokenizer.eos_token_id:
                encoded_example['preamble']['input_ids'] = encoded_example['preamble']['input_ids'][:-1]

            encoded_example['context'] = self.tokenizer(ctxt, add_special_tokens=False)

            encoded_example['aliases'] = list(self.samples[sample_idx]['aliases'])
            encoded_example['cot_delimiter'] = cot_delimiter
            examples.append(encoded_example)
            for answer in self.samples[sample_idx]['aliases']:
                response = f"{self.samples[sample_idx]['chain_of_thought']}{cot_delimiter}{answer}"
                max_answer_length = max(max_answer_length, len(self.tokenizer(response)['input_ids']))

            if len(self.samples[sample_idx]['chain_of_thought']) > 0:
                has_cot = True

        self.max_answer_length = max_answer_length + (_MAX_ANSWER_BUFFER_LENGTH if has_cot else 0)
        return examples

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

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

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0]

    def split_batch(self, batch: Any, microbatch_size: int):
        # Don't split kwargs that don't change
        # Normally split torch tensors
        # List split lists of strings
        no_split = ['mode', 'generation_length', 'generation_kwargs', 'cot_delimiter']
        normal_split = ['input_ids', 'attention_mask']
        list_split = ['labels']
        chunked = {}
        for k, v in batch.items():
            if k in no_split:
                # Defer broadcasting until we know num_chunks
                pass
            elif k in list_split:
                chunked[k] = _split_list(v, microbatch_size)
            elif k in normal_split:
                chunked[k] = _default_split_batch(v, microbatch_size)
            else:
                raise ValueError(f'Unexpected key {k}')
        num_chunks = len(chunked['input_ids'])
        for k, v in batch.items():
            if isinstance(v, (int, float, str, bool, dict)):
                chunked[k] = [v] * num_chunks
        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


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
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
        fewshot_random_seed (int): Random seed used to select fewshot examples
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
        fewshot_random_seed: int,
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
        self.samples = list(
            dataset.map(lambda examples: {
                'continuation': examples['continuation'],
                'context': examples['context'],
            }))
        self.samples = strip_data(self.samples)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        fewshot_rng = random.Random(fewshot_random_seed)

        self.prefix_space = _tokenizer_needs_prefix_space(self.tokenizer)

        self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter, continuation_delimiter,
                                                  fewshot_rng)

    def prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, continuation_delimiter: str,
                      fewshot_rng: random.Random):
        """Prepares a set of language modeling tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            continuation_delimiter (str): The delimiter used to separate each context from its continuation
            fewshot_rng (random.Random): Random number generator used to select fewshot examples

        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):
            encoded_example = {}

            preamble = prompt_string

            if num_fewshot > 0:
                fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx, fewshot_rng)
                for fewshot_idx in fewshot_idxs:
                    ctxt, cont = self.samples[fewshot_idx]['context'], self.samples[fewshot_idx]['continuation']
                    if len(preamble) > 0:
                        ctxt = f'{example_delimiter}{ctxt}'
                    preamble += f'{ctxt}{continuation_delimiter}{cont}'

            ctxt, cont = self.samples[sample_idx]['context'], self.samples[sample_idx]['continuation']
            if len(preamble) > 0:
                ctxt = f'{example_delimiter}{ctxt}'

            # rstrip the continuation delimiter, because the prompt ending in a space results in degenerate output
            continuation_delimiter_stripped = continuation_delimiter.rstrip()

            if self.prefix_space and not cont.startswith(' '):
                cont = f' {cont}'
            ctxt += continuation_delimiter_stripped

            encoded_example['preamble'] = self.tokenizer(
                preamble
            )  # if the preamble is empty then these will be 0-length lists, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
            if self.tokenizer.eos_token_id is not None and len(
                    encoded_example['preamble']
                ['input_ids']) > 1 and encoded_example['preamble']['input_ids'][-1] == self.tokenizer.eos_token_id:
                encoded_example['preamble']['input_ids'] = encoded_example['preamble']['input_ids'][:-1]

            encoded_example['context'] = self.tokenizer(ctxt, add_special_tokens=False)
            encoded_example['continuation'] = self.tokenizer(cont, add_special_tokens=False)

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
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
        fewshot_random_seed (int): Random seed used to select fewshot examples
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
        fewshot_random_seed: int,
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
        self.samples = list(
            dataset.map(lambda examples: {
                'query': examples['query'],
                'choices': examples['choices'],
                'gold': examples['gold']
            }))
        self.samples = strip_data(self.samples)

        self.num_choices = len(self.samples[0]['choices'])
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        fewshot_rng = random.Random(fewshot_random_seed)

        self.prefix_space = _tokenizer_needs_prefix_space(self.tokenizer)

        self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter, continuation_delimiter,
                                                  fewshot_rng)

    def prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, continuation_delimiter: str,
                      fewshot_rng: random.Random):
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
            fewshot_rng (random.Random): Random number generator used to select fewshot examples

        Returns:
            dict: Contains the query, the list of encoded potential answer choices, the preamble (prompt + fewshot examples), and
                the index of the correct answer choice.
        """
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):

            preamble = prompt_string
            if num_fewshot > 0:
                fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx, fewshot_rng)
                for fewshot_idx in fewshot_idxs:
                    query, choices, gold_idx = self.samples[fewshot_idx]['query'], self.samples[fewshot_idx][
                        'choices'], self.samples[fewshot_idx]['gold']
                    if len(preamble) > 0:
                        query = f'{example_delimiter}{query}'
                    assert isinstance(gold_idx, int)
                    preamble += f'{query}{continuation_delimiter}{choices[gold_idx]}'
            encoded_example = {}
            query, choices, gold_idx = self.samples[sample_idx]['query'], self.samples[sample_idx][
                'choices'], self.samples[sample_idx]['gold'],
            if len(preamble) > 0:
                query = f'{example_delimiter}{query}'

            # rstrip the continuation delimiter, because the prompt ending in a space results in degenerate output
            continuation_delimiter_stripped = continuation_delimiter.rstrip()

            if self.prefix_space:
                choices = [(f' {choice}' if not choice.startswith(' ') else choice) for choice in choices]
            query += continuation_delimiter_stripped
            encoded_example['preamble'] = self.tokenizer(
                preamble
            )  # if the preamble is empty then these will be 0-length lists, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)

            if self.tokenizer.eos_token_id is not None and len(
                    encoded_example['preamble']
                ['input_ids']) > 1 and encoded_example['preamble']['input_ids'][-1] == self.tokenizer.eos_token_id:
                encoded_example['preamble']['input_ids'] = encoded_example['preamble']['input_ids'][:-1]

            encoded_example['gold_idx'] = gold_idx

            encoded_example['query'] = self.tokenizer(query, add_special_tokens=False)
            encoded_example['choices'] = [self.tokenizer(choice, add_special_tokens=False) for choice in choices]

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
        # Don't split kwargs that don't change
        # Normally split torch tensors
        # List split lists of strings
        no_split = ['mode']
        # Real
        real = ['input_ids', 'labels', 'attention_mask']
        logical = ['gold_indices']
        chunked = {}
        for k, v in batch.items():
            if k in no_split:
                # Defer broadcasting primitives until we know num_chunks
                pass
            elif k == 'continuation_indices':
                # List of list, so we have to directly call _split_list
                chunked[k] = _split_list(v, microbatch_size * self.num_choices)
            elif k == 'choice_groupings':
                # List of list, so we have to directly call _split_list
                chunked[k] = _split_list(v, microbatch_size)
            elif k in real:
                chunked[k] = _default_split_batch(v, microbatch_size * self.num_choices)
            elif k in logical:
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
        self.samples = list(
            dataset.map(
                lambda examples: {
                    'context_options': examples['context_options'],
                    'continuation': examples['continuation'],
                    'gold': examples['gold']
                }))
        self.samples = strip_data(self.samples)

        self.num_choices = len(self.samples[0]['context_options'])
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        fewshot_rng = random.Random(fewshot_random_seed)

        self.prefix_space = _tokenizer_needs_prefix_space(self.tokenizer)

        self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter, continuation_delimiter,
                                                  fewshot_rng)

    def prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, continuation_delimiter: str,
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

        examples = []
        for sample_idx in tqdm(range(len(self.samples))):

            preamble = prompt_string
            if num_fewshot > 0:
                fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx, fewshot_rng)
                for fewshot_idx in fewshot_idxs:
                    context_options, continuation, gold_idx = self.samples[fewshot_idx][
                        'context_options'], self.samples[fewshot_idx]['continuation'], self.samples[fewshot_idx]['gold']
                    assert isinstance(gold_idx, int)
                    context = context_options[gold_idx]
                    if len(preamble) > 0:
                        context = f'{example_delimiter}{context}'
                    preamble += f'{context}{continuation_delimiter}{continuation}'

            encoded_example = {}
            context_options, continuation, gold_idx = self.samples[sample_idx]['context_options'], self.samples[
                sample_idx]['continuation'], self.samples[sample_idx]['gold'],

            # rstrip the continuation delimiter, because the prompt ending in a space results in degenerate output
            continuation_delimiter_stripped = continuation_delimiter.rstrip()

            if len(preamble) > 0:
                context_options = [f'{example_delimiter}{c}{continuation_delimiter_stripped}' for c in context_options]
            encoded_example['preamble'] = self.tokenizer(
                preamble
            )  # if the preamble is empty then these will be 0-length lists, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
            if self.tokenizer.eos_token_id is not None and len(
                    encoded_example['preamble']
                ['input_ids']) > 1 and encoded_example['preamble']['input_ids'][-1] == self.tokenizer.eos_token_id:
                encoded_example['preamble']['input_ids'] = encoded_example['preamble']['input_ids'][:-1]

            encoded_example['gold_idx'] = gold_idx
            encoded_example['context_options'] = [self.tokenizer(c, add_special_tokens=False) for c in context_options]

            if self.prefix_space:
                continuation = f' {continuation}' if not continuation.startswith(' ') else continuation
            encoded_example['continuation'] = self.tokenizer(continuation, add_special_tokens=False)
            examples.append(encoded_example)

        return examples

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


class InContextLearningCodeEvalDataset(Dataset):
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

    def __init__(
        self,
        dataset_uri: str,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        max_seq_len: int,
        pad_tok_id: int,
        num_fewshot: int,
        prompt_string: str,
        example_delimiter: str,
        destination_path: str,
        code_prelimiter: str,
        fewshot_random_seed: int,
        generations_per_sample: int,
        pass_at_k: int = 1,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = 40,
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
        self.samples = list(
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

        if generations_per_sample < pass_at_k:
            raise ValueError(
                f'generations_per_sample ({generations_per_sample}) must be greater than or equal to pass_at_k ({pass_at_k}) for code evaluation.'
            )

        self.pass_at_k = pass_at_k
        self.generations_per_sample = generations_per_sample

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.padding_side = 'left'
        self.max_prompt_length = 0
        self.top_p = top_p
        self.top_k = top_k
        fewshot_rng = random.Random(fewshot_random_seed)
        self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter, code_prelimiter,
                                                  fewshot_rng)

    def prep_examples(self, num_fewshot: int, prompt_string: str, example_delimiter: str, code_prelimiter: str,
                      fewshot_rng: random.Random):
        """Prepares a set of code evaluation tasks into tokenized format with prompt and fewshot examples.

        Each task consists of a context as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            example_delimiter (str): The delimiter used to separate each individual context/continuation pair
            code_prelimiter (str): The text to prepend to each code prompt
            fewshot_rng (random.Random): Random number generator to use for fewshot sampling

        Returns:
            dict: Contains the context, the continuation, and the preamble (prompt + fewshot examples)
        """
        max_prompt_length = 0
        examples = []
        for sample_idx in tqdm(range(len(self.samples))):
            encoded_example = {}

            preamble = prompt_string

            if num_fewshot > 0:
                fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx, fewshot_rng)
                for fewshot_idx in fewshot_idxs:
                    ctxt, cont = self.samples[fewshot_idx]['prompt'], self.samples[fewshot_idx]['canonical_solution']
                    ctxt = f'{code_prelimiter}{ctxt}'
                    if len(preamble) > 0:
                        ctxt = f'{example_delimiter}{ctxt}'
                    preamble += f'{ctxt}{cont}'

            ctxt = self.samples[sample_idx]['prompt']
            ctxt = f'{code_prelimiter}{ctxt}'
            if len(preamble) > 0:
                ctxt = f'{example_delimiter}{ctxt}'

            # If the preamble is empty then this will be a 0-length list, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
            encoded_example['preamble'] = self.tokenizer(preamble)
            # If there is an EOS token added, we need to remove it so it is not in the middle of the prompt
            if self.tokenizer.eos_token_id is not None and len(
                    encoded_example['preamble']
                ['input_ids']) > 1 and encoded_example['preamble']['input_ids'][-1] == self.tokenizer.eos_token_id:
                encoded_example['preamble']['input_ids'] = encoded_example['preamble']['input_ids'][:-1]

            encoded_example['prompt'] = self.tokenizer(ctxt, add_special_tokens=False)
            encoded_example['prompt_text'] = self.samples[sample_idx]['prompt']
            encoded_example['task_id'] = self.samples[sample_idx]['task_id']
            encoded_example['canonical_solution'] = self.samples[sample_idx]['canonical_solution']
            encoded_example['test'] = self.samples[sample_idx]['test']
            encoded_example['entry_point'] = self.samples[sample_idx]['entry_point']
            encoded_example['test_inputs'] = self.samples[sample_idx]['test_inputs']
            encoded_example['test_outputs'] = self.samples[sample_idx]['test_outputs']
            encoded_example['language'] = self.samples[sample_idx]['language']

            examples.append(encoded_example)
            max_prompt_length = max(
                max_prompt_length,
                len(encoded_example['preamble']['input_ids'] + encoded_example['prompt']['input_ids']))

        self.max_prompt_length = max_prompt_length
        return examples

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

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

    def get_num_samples_in_batch(self, batch) -> int:
        # Count number of inputs in the batch
        return batch['input_ids'].shape[0]

    def split_batch(self, batch: Any, microbatch_size: int):
        # Don't split kwargs that don't change
        # Normally split torch tensors
        # List split lists of strings
        no_split = ['mode', 'generation_length', 'pass_at_k', 'generation_kwargs']
        normal_split = ['input_ids', 'attention_mask']
        list_split = [
            'labels', 'tests', 'canonical_solutions', 'entry_points', 'test_inputs', 'test_outputs', 'prompts',
            'languages'
        ]
        chunked = {}
        for k, v in batch.items():
            if k in no_split:
                # Defer broadcasting until we know num_chunks
                pass
            elif k in list_split:
                chunked[k] = _split_list(v, microbatch_size)
            elif k in normal_split:
                chunked[k] = _default_split_batch(v, microbatch_size)
            else:
                raise ValueError(f'Unexpected key {k}')
        num_chunks = len(chunked['input_ids'])
        for k, v in batch.items():
            if isinstance(v, (int, float, str, bool, dict)):
                chunked[k] = [v] * num_chunks

        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


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
    question_prelimiter: str = '',  # e.g. 'Question: '
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
                                                 question_prelimiter=question_prelimiter,
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
                                                   code_prelimiter=question_prelimiter,
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
        question_prelimiter: str = '',  # e.g. 'Question: '
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
        question_prelimiter: (str): For QA tasks, this will be prepended to each question.
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
                question_prelimiter,
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
            question_prelimiter,
            cot_delimiter,
            fewshot_random_seed,
            pass_at_k,
            generations_per_sample,
        )
