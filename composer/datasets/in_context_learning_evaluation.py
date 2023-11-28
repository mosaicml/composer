# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from composer.core import DataSpec
from composer.core.data_spec import _default_split_batch, _split_list
from composer.utils import MissingConditionalImportError, dist, get_file, parse_uri

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


def _check_if_huggingface_uri(uri: str) -> bool:
    """
    Takes a dataset uri and checks if it's a HuggingFace dataset uri.
    Returns False if a backend uri is present (ie 's3://', 'oci://') or if the uri is a local file.
    Returns True otherwise.
    Args:
        uri (str): uri as a string

    Returns:
        bool: result of parsing uri as a HF uri
    """
    backend, _, path = parse_uri(uri)
    if backend == '':
        _, ext = os.path.splitext(path)
        # If there's any extension, it's a link to a local file. If no extention, HF path
        return ext == ''
    # If there's any backend, it's a cloud OCI and not HF
    return False


def strip_data(sample: Dict) -> Dict:
    return {k: v.strip() if isinstance(v, str) else v for k, v in sample.items()}


def _tokenizer_needs_prefix_space(tokenizer: transformers.PreTrainedTokenizerBase) -> bool:
    """
    Test for whether a prefix space is needed before the continuation.
    Sentencepiece tokenization should not have a prefix space, but gpt2 style BPE should.
    """
    return len(tokenizer(' a', add_special_tokens=False)['input_ids']) == 1


def _make_padded_input(context_enc: List,
                       continuation_enc: List,
                       max_seq_len: int,
                       pad_tok_id: int,
                       padding_side: str = 'right') -> Tuple[torch.tensor, torch.tensor]:
    # TODO: docstring
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


def _get_fewshot_sample_idxs(dataset_size: int, num_fewshot: int, sample_idx: int, rng: random.Random) -> List[int]:
    """
    Samples without replacement. If num_fewshot exceeds the number of unique samples,
    then we will have fewer than num_fewshot examples in context.
    Args:
        dataset_size (int): length of the dataset
        num_fewshot (int): number of examples to prepend
        sample_idx (int): current sample index (excluded from fewshot choices)
        rng (random.Random): rng for repeatable sample selection

    Returns:
        list: indices of the examples chosen for fewshot selection
    """
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
    """
    A base dataset that constructs batches for in-context learning task evaluations

    The input format is expected to be a jsonl file with different fields based on the task or a link to a Hugging Face dataset.

    Args:
        dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri.
            Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            A local dataset must consist of rows of JSON data points with different fields based on the task.
            The default keys expected are "context" and "answer".
        tokenizer (transformers.PreTrainedTokenizerBase): The tokenizer used to map between strings and token ids
        max_seq_len (int): The maximum sequence length supported by the model
        pad_tok_id (int): The special token reserved for padding batches
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example
        fewshot_random_seed (int): Random seed to use for fewshot sampling
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual (context, answer) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and answer in each example (e.g. '\nA: ')
        prelimiter (str): Text to be prepended before each example, including few shot examples
        context_key (str): The key from the parsed dataset that the class will use as the "context" (i.e. the main content to be included in the prompt)
        answer_key (str): The key from the parsed dataset that the class will use as the "answer" (i.e. the main content to be predicted by the model)
        destination_path (str): Temporary path to store downloaded datasets
        strip_dataset (bool): Boolean for whether to strip whitespace from data. Trailing whitespace can cause degenerative outputs,
            so unless whitespace should be preserved (for example in code), this should be set to True.
        hf_loading_vars (Dict): A dictionary containing keyword arguments to be passed into `load_dataset` if dataset is being pulled from HF.
        hf_parsing_map (Dict[str:List[str]]): A dictionary containing a mapping from HF columns to ICL dataset keys. The dictionary should be formatted {icl_key:[hf_key1, hf_key1]}.
            Values in the dict will be concatenated with ' ' seperating them. If not included, will use the columns already present in the HF dataset.
        stacked_keys (List(str)): keys in the output batch that must be converted to tensors with torch.stack()
        dont_split_keys (List(str)): keys in the ICL dictionary that should not be split among batches.
        list_split_keys (List(str)): keys in the ICL dictionary that will be split as lists, resulting in microbatch_size sections of the list being inserted in every batch
        normal_split_keys (List(str)): keys in the ICL dictionary that will be split into chunks regularly
    """

    def __init__(self,
                 dataset_uri: str,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_seq_len: int,
                 pad_tok_id: int,
                 num_fewshot: int,
                 fewshot_random_seed: int,
                 prompt_string: str,
                 example_delimiter: str,
                 continuation_delimiter: str,
                 destination_path: str,
                 prelimiter: str = '',
                 context_key: str = 'context',
                 answer_key: str = 'answer',
                 strip_dataset: bool = True,
                 hf_loading_vars: Dict = None,
                 hf_parsing_map: Dict = None,
                 stacked_keys: List[str] = None,
                 dont_split_keys: List[str] = None,
                 list_split_keys: List[str] = None,
                 normal_split_keys: List[str] = None):
        self.tokenizer = tokenizer
        self.prefix_space = _tokenizer_needs_prefix_space(self.tokenizer)

        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.num_fewshot = num_fewshot
        # TODO: check this is correct for all dataset types
        self.padding_side = 'left'

        self.prelimiter = prelimiter
        self.example_delimiter = example_delimiter
        self.continuation_delimiter = continuation_delimiter
        self.context_key = context_key
        self.answer_key = answer_key
        self.stacked_keys = stacked_keys or ['input_ids', 'labels']
        self.dont_split_keys = dont_split_keys or []
        self.list_split_keys = list_split_keys or []
        self.normal_split_keys = normal_split_keys or []

        hf_loading_vars = hf_loading_vars or {}
        self.dataset = self._read_dataset(dataset_uri, destination_path, hf_loading_vars, hf_parsing_map)
        self.strip_data = strip_dataset
        if self.strip_data:
            self.dataset = self.dataset.map(strip_data)

        fewshot_rng = random.Random(fewshot_random_seed)
        self.encoded_dataset = self.dataset.map(
            self._prep_example,
            with_indices=True,
            fn_kwargs={
                'num_fewshot': num_fewshot,
                'prompt_string': prompt_string,
                'fewshot_rng': fewshot_rng,
            },
        )

    def __getitem__(self, index: int) -> Dict:
        return self.encoded_dataset[index]

    def __len__(self) -> int:
        return len(self.encoded_dataset)

    def get_num_samples_in_batch(self, batch: Dict) -> int:
        return batch['input_ids'].shape[0]

    def check_defaults_are_set(self, dict_of_defaults: dict) -> None:
        if all(v for v in dict_of_defaults.values()):
            return
        raise ValueError(
            f"{type(self).__name__} missing required variable(s): {', '.join([k for k, v in dict_of_defaults.items() if not v])}"
        )

    def _read_dataset(self,
                      dataset_uri: str,
                      destination_path: str,
                      hf_loading_vars: Dict = None,
                      hf_parsing_map: Dict = None) -> transformers.Dataset:
        """
        Reads a dataset and handles parsing it from HuggingFace.
        Args:
            dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri.
                Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            destination_path (str): A local path where the data will be stored
            hf_loading_vars (Dict): If parsing from HuggingFace, keyword args that will be passed into load_dataset
            hf_parsing_map (Dict): Dictionary in the form of {icl_key: [hf_col1, hf_col2]} that will map one or more hf columns, in order, to ICL dataset columns

        Returns:
            dataset: a loaded HF dataset
        """
        try:
            from datasets import load_dataset  # pyright: ignore [reportGeneralTypeIssues]
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='nlp',
                conda_package='datasets',
                conda_channel='conda-forge',
            ) from e
        if _check_if_huggingface_uri(dataset_uri):
            dataset = load_dataset(dataset_uri, **hf_loading_vars)
            if hf_parsing_map:
                dataset_parsing_func = lambda example: {
                    k: ' '.join([str(example[col]) for col in v]) for k, v in hf_parsing_map.items()
                }
                dataset = dataset.map(dataset_parsing_func, remove_columns=dataset.column_names)
        else:
            with dist.local_rank_zero_download_and_wait(destination_path):
                if dist.get_local_rank() == 0:
                    get_file(dataset_uri, destination_path, overwrite=True)
            dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        return dataset

    def _generate_few_shot_text(
        self,
        num_fewshot: int,
        sample_idx: int,
        preamble: str,
        fewshot_rng: random.Random,
    ) -> str:
        """
        Formats the prompt fewshot examples for test sample `sample_idx`.

        Randomly select `num_fewshot` samples from the dataset (not including the sample at `sample_idx`) and constructs
        a context with its answer appended.

        Returns the formatted prompt_string + concatenated list of formatted few shot examples.
        Args:
            num_fewshot (int): number of examples to prepend
            sample_idx (int): current sample idx
            preamble (str): text to occur at the beginning of the task. Generally instructions or a prompt.
            fewshot_rng (random.Random): seeded sampler to chose samples with

        Returns:
            str: the original preamble with num_fewshot examples appended
        """
        few_shot_text = preamble

        if num_fewshot > 0:
            fewshot_idxs = _get_fewshot_sample_idxs(len(self.dataset), num_fewshot, sample_idx, fewshot_rng)
            for fewshot_idx in fewshot_idxs:
                ctxt = self._construct_context(self.dataset[fewshot_idx], few_shot_text, add_answer=True)
                few_shot_text += ctxt

        return few_shot_text

    def _construct_context(self, sample: Dict, preceding_text: str = '', add_answer: bool = False) -> str:
        """
        Takes a sample and constructs a context. Optionally adds the correct answer (for fewshot examples)
        and handle exmple delemiters

        Args:
            sample (Dict): the sample from which to construct the context
            preceding_text (str): any preceding text, used as a check for prepending self.example_delimiter
            add_answer (bool): bool for whether or not to add the answer on the end of the context (e.g. for fewshot examples)

        Returns:
            str: The constructed context. The default output context is
                 formatted as follows: f'{self.prelimiter}{sample[self.context_key]}{self.continuation_delimiter}'
        """
        ctxt = sample[self.context_key]
        ctxt = f'{self.prelimiter}{ctxt}'
        if len(preceding_text) > 0:
            ctxt = f'{self.example_delimiter}{ctxt}'
        ctxt = f'{ctxt}{self.continuation_delimiter}'
        if add_answer:
            ctxt = f'{ctxt}{self._get_answer_from_sample(sample)}'
        return ctxt

    def _get_answer_from_sample(self, sample: Dict[str, Any]) -> str:
        """
        Returns the answer from the sample
        Args:
            sample (Dict): the sample from which to retrieve the answer

        Returns:
            str: the answer in the sample
        """
        return sample[self.answer_key]

    def _fix_eos_on_preamble(self, input_ids: List[int]) -> List[int]:
        """
        If the input_ids is empty then input_ids['input_ids'] will be a 0-length List,
        unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
        If there is an EOS token added, we need to remove it so it is not in the middle of the prompt,
        as the specific eval question's prompt will follow theinput_ids
        Args:
            input_ids (List): the tokenized input

        Returns:
            input_ids: the tokenized input conditionally edited
        """
        if (self.tokenizer.eos_token_id is not None and len(input_ids) > 1 and
                input_ids[-1] == self.tokenizer.eos_token_id):
            input_ids = input_ids[:-1]
        return input_ids

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handles special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctxt (str): the specific example's derrived context
            example (Dict): the example as a dictionary. Used for additional processing in inherited classes.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = {}
        preamble = self.tokenizer(prompt_and_fewshot)
        preamble['input_ids'] = self._fix_eos_on_preamble(preamble['input_ids'])
        tokenized_example['preamble'] = preamble
        if self.strip_data:
            # rstrip context because a prompt ending in a space results in degenerate output
            ctxt = ctxt.rstrip()
        tokenized_example['context'] = self.tokenizer(ctxt, add_special_tokens=False)
        return tokenized_example

    def _prep_example(
        self,
        example: Dict,
        example_idx: int,
        num_fewshot: int,
        prompt_string: str,
        fewshot_rng: random.Random,
    ) -> List[Dict[str, Any]]:
        """
        Prepares a single example from a HF Dataset into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            example (Dict): A Dictionary from the hf dataset
            example_idx (int): the index of example
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            fewshot_rng (random.Random): Random number generator to use for fewshot sampling

        Returns:
            Dict: contains a dictionary with the tokenized data
        """
        prompt_and_fewshot = self._generate_few_shot_text(num_fewshot, example_idx, prompt_string, fewshot_rng)
        ctxt = self._construct_context(example, prompt_and_fewshot, add_answer=False)
        tokenized_example = self._tokenize_example(prompt_and_fewshot, ctxt, example)
        return tokenized_example

    def collate_fn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.
        Args:
            data (List): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: dictionary for a single batch
        """
        batch = {
            'input_ids': [],
            'continuation_indices': [],
            'mode': 'icl_task',
            'labels': [],
        }
        for data_pair in data:
            context_enc = data_pair['preamble']['input_ids'] + data_pair['context']['input_ids']

            inp, continuation_span = _make_padded_input(context_enc, data_pair['continuation']['input_ids'],
                                                        self.max_seq_len, self.pad_tok_id)

            batch['input_ids'].append(inp)
            batch['continuation_indicies'].append(continuation_span)
            batch['labels'].append(inp)

        batch = {k: torch.stack(v) if k in self.stacked_keys else v for k, v in batch.items()}
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch

    def split_batch(self, batch: Any, microbatch_size: int) -> List[Dict[str, Any]]:
        """
        Handling for certain specialty columns that must be split into batches in different formats.

        Args:
            batch (Dict): batch of data
            microbatch_size (int): size of microbatches

        Returns:
            List: list of chunked batches
        """
        # Don't split kwargs that don't change
        # Normally split torch tensors
        # List split lists of strings
        self.check_defaults_are_set({
            'dont_split_keys': self.dont_split_keys,
            'list_split_keys': self.list_split_keys,
            'normal_split_keys': self.normal_split_keys
        })
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
            if isinstance(v, (int, float, str, bool, Dict)):
                chunked[k] = [v] * num_chunks

        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


class InContextLearningRAGGenerationTaskDataset(InContextLearningDataset):
    """A dataset that construct batches for in-context learning RAG generation evaluation
    Rag generation tasks evaluate a model's ability to answer questions based on passages.

    Args:
        passage_delimiter (str): Delimiter to place between each passage.
        passage_query_delimiter (str): Delimiter to place between the last passage and the query.
    """

    def __init__(
            self,
            passage_delimiter: str = '\nPassage: ',
            passage_query_delimiter: str = '\nQuery: ',
            *args,
            **kwargs
            ):
        kwargs.pop('passage_delimiter', None)
        kwargs.pop('passage_query_delimiter', None)
        self.passage_delimiter = passage_delimiter
        self.passage_query_delimiter = passage_query_delimiter
        super().__init__(*args, **kwargs)

    def _construct_context(self, sample: dict, preceding_text: str = '', add_answer: bool = False):
        """
        Takes a sample and constructs a context. Optionally, appends this to preceeding text (such as a
        prompt or fewshot examples), as well as optionally adds the correct answer (for fewshot examples)

        Args:
            sample (dict): the sample from which to construct the context
            preceding_text (str): any preceding text, needed to if self.example_delimiter is needed at the beginning
            add_answer (bool): bool for whether or not to add the answer on the end of the context (needed for fewshot examples)

        Returns:
            str: The constructed context. The default output context is
                 formatted as follows: f'{self.prelimiter}{sample['self.passages_key']}{sample[self.context_key]}{self.continuation_delimiter}'
        """
        passages = self.passage_delimiter.lstrip('\n ')
        passages += f'{self.passage_delimiter}'.join(sample['passages'])
        query = sample['query']
        context = f'{self.prelimiter}{passages}{self.passage_query_delimiter}{query}'
        return context

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: dict):
        """
        Runs text through the tokenizer and handles special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derived context
            example (dict): the example as a dictionary.

        Returns:
            dict: dictionary with the tokenized data
        """
        tokenized_example = super()._tokenize_example(prompt_and_fewshot, ctxt, example)
        answer = example['answers'][0]
        tokenized_example['answer'] = self.tokenizer(answer, add_special_tokens=False)
        return tokenized_example


    def collate_fn(self, data):
        """
        The function that the dataloader uses to accumulate data into batches
        Args:
            data (list): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            dict: dictionary for a single batch
        """
        batch = {
            'input_ids': [],
            'continuation_indices': [],
            'mode': 'icl_task',
            'labels': [],
            'answer_indices': []
            }
        for data_pair in data:
            context_enc = data_pair['preamble']['input_ids'] + data_pair['context']['input_ids']
            answer_enc = data_pair['answer']['input_ids']

            inp, answer_span = _make_padded_input(context_enc, answer_enc, self.max_seq_len,
                                                        self.pad_tok_id)
            batch['input_ids'].append(inp)
            batch['answer_indices'].append(answer_span)
            batch['labels'].append(inp)

        batch = {k: torch.stack(v) if k in self.stacked_keys else v for k, v in batch.items()}
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch


class InContextLearningQATaskDataset(InContextLearningDataset):
    """
    A dataset that construct batches for in-context learning question answering evaluation.
    QA tasks evaluate a model's ability to answer questions using a consistent format.

    The input format is expected to be a jsonl file with the following fields:
    - context: the question
    - answer: the preferred answer to the question
    - aliases: a list of aliases for the answer

    See InContextLearningDataset for more details.

    Additional Args:
        cot_delimiter (str): Delimiter to place between the chain of thought and continuations.
    """

    def __init__(self, cot_delimiter: str = '', *args, **kwargs):
        self.cot_delimiter = cot_delimiter
        self.has_cot = False
        super().__init__(stacked_keys=['input_ids'],
                         dont_split_keys=['mode', 'generation_length', 'generation_kwargs', 'cot_delimiter'],
                         normal_split_keys=['input_ids', 'attention_mask'],
                         list_split_keys=['labels'],
                         *args,
                         **kwargs)

        self.max_answer_length = self.get_max_answer_length()

    def _read_dataset(
        self,
        dataset_uri: str,
        destination_path: str,
        hf_loading_vars: Dict = None,
        hf_parsing_map: Dict = None,
    ):
        dataset = super()._read_dataset(dataset_uri, destination_path, hf_loading_vars, hf_parsing_map)
        self.has_cot = 'chain_of_thought' in dataset.features
        return dataset.map(
            lambda examples: {
                'context': examples['context'],
                'answer': examples['answer'],
                'aliases': set([examples['answer']] + examples.get('aliases', [])),
                'chain_of_thought': examples.get('chain_of_thought', ''),
            })

    def _get_answer_from_sample(self, sample: Dict) -> str:
        """
        Returns the answer from the sample. Applies chain of thought if self.has_cot is marked as true.
        Args:
            sample (Dict): the sample from which to retrieve the answer

        Returns:
            str: the answer in from the sample with chain of thought and delimiter if needed
        """
        if self.has_cot:
            return f'{sample["chain_of_thought"]}{self.cot_delimiter}{sample[self.answer_key]}'
        else:
            return sample[self.answer_key]

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handles special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derrived context
            example (Dict): the example as a dictionary.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = super()._tokenize_example(prompt_and_fewshot, ctxt, example)
        tokenized_example['aliases'] = list(example.get('aliases', []))
        return tokenized_example

    def get_max_answer_length(self) -> int:
        f"""
        Loops over the dataset and finds the longest answer length.

        Returns:
            int: the maximum answer length with an additional buffer of {_MAX_ANSWER_BUFFER_LENGTH} if chain of thought is present
        """
        max_answer_length = 0
        for sample in self.dataset:
            all_answers = [sample[self.answer_key]] + list(sample.get('aliases', []))
            for answer in all_answers:
                if self.has_cot:
                    response = (f'{sample["chain_of_thought"]}{self.cot_delimiter}{answer}')
                else:
                    response = answer
                max_answer_length = max(max_answer_length, len(self.tokenizer(response)['input_ids']))
        max_answer_length = max_answer_length + (_MAX_ANSWER_BUFFER_LENGTH if len(self.cot_delimiter) > 0 else 0)
        return max_answer_length

    def collate_fn(self, data: Dict) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.
        Args:
            data (List): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: dictionary for a single batch
        """
        batch = {
            'input_ids': [],
            'mode': 'generate',
            'labels': [],
            'cot_delimiter': self.cot_delimiter,
            'generation_length': self.max_answer_length,
            'generation_kwargs': {
                'pad_token_id': self.pad_tok_id,
                'use_cache': True
            },
        }
        for sample in data:
            aliases = sample['aliases']
            context_enc = sample['preamble']['input_ids'] + sample['context']['input_ids']
            inp, _ = _make_padded_input(
                context_enc,
                [],
                self.max_seq_len - self.max_answer_length,
                self.pad_tok_id,
                padding_side=self.padding_side,
            )

            batch['input_ids'].append(inp)
            batch['labels'].append(aliases)

        batch = {k: torch.stack(v) if k in self.stacked_keys else v for k, v in batch.items()}
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch


class InContextLearningLMTaskDataset(InContextLearningDataset):
    """
    A dataset that construct batches for in-context learning language modeling evaluation.
    Language modeling tasks test a model's ability to properly predict tokens based on preceding tokens.

    The input format is expected to be a jsonl file with the following fields:
    - context: preceding text
    - continuation: the expected continuation

    See InContextLearningDataset for more details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(answer_key='continuation', *args, **kwargs)

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handles special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derrived context
            example (Dict): the example as a dictionary.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = super()._tokenize_example(prompt_and_fewshot, ctxt, example)
        cont = example['continuation']
        if self.prefix_space and not cont.startswith(' '):
            cont = f' {cont}'
        tokenized_example['continuation'] = self.tokenizer(cont, add_special_tokens=False)
        return tokenized_example

    def collate_fn(self, data: Dict[str, any]) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.
        Args:
            data (List): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: dictionary for a single batch
        """
        batch = {'input_ids': [], 'continuation_indices': [], 'mode': 'icl_task', 'labels': []}
        for data_pair in data:
            context_enc = data_pair['preamble']['input_ids'] + data_pair['context']['input_ids']
            continuation_enc = data_pair['continuation']['input_ids']

            inp, continuation_span = _make_padded_input(context_enc, continuation_enc, self.max_seq_len,
                                                        self.pad_tok_id)
            batch['input_ids'].append(inp)
            batch['continuation_indices'].append(continuation_span)
            batch['labels'].append(inp)

        batch = {k: torch.stack(v) if k in self.stacked_keys else v for k, v in batch.items()}
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch


class InContextLearningMultipleChoiceTaskDataset(InContextLearningDataset):
    """
    A dataset that construct batches for in-context learning multiple choice evaluation.

    If each question has N answer choices, we construct N distinct inputs per question. In order to ensure
    consistency across multi-GPU, we set the batch size to be `min(N, batch_size)` so that all N
    inputs per question can stored in the same batch.

    The default input format is a jsonl file with the following fields:
    - query: the preceding text, question, or document relevant to the choices
    - gold: index of the correct choice under 'choices'
    - choices: a list of strings, each being one of the potential choices

    Each batch then consists of batch_size // N distinct questions and has the following the structure.
    - input_ids: Input tensor batch x seqlen x # tokens
    - continuation_indices: List of |batch| consisting of tensors indicating which indices in the sequence correspond to the question answer (aka continuation)
    - mode: Indicates to the model that this is an ICL task and may rely on a custom code path to properly update metrics
    - labels: Identical to the input, used by the model to calculate loss/metrics
    - gold_indices: List of length |batch_size // N| indicating for each question, which of the answers is correct (via an integer [0, N-1])
    - choice_groupings: Indicates which indices of the batch correspond to which questions

    Additional Args:
        choices_key (str): the key under which the choices are stored in the saved dataset. Defaults to 'choices'.
    """

    def __init__(self, choices_key: str = 'choices', *args, **kwargs):
        super().__init__(context_key='query',
                         dont_split_keys=['mode'],
                         normal_split_keys=['gold_indices'],
                         *args,
                         **kwargs)
        self.num_choices = len(self.dataset[0][choices_key])
        self.real_split_keys = ['input_ids', 'labels', 'attention_mask']

    def _get_answer_from_sample(self, sample: Dict) -> str:
        """
        Returns the correct answer from the sample's choices.
        Args:
            sample (Dict): the sample from which to retrieve the answer

        Returns:
            str: the full string of the correct answer based on the 'gold' key
        """
        choices = sample['choices']
        gold_idx = sample['gold']
        return choices[gold_idx]

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handles special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derrived context
            example (Dict): the example as a dictionary.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = super()._tokenize_example(prompt_and_fewshot, ctxt, example)
        choices = example['choices']
        if self.prefix_space:
            choices = [(f' {choice}' if not choice.startswith(' ') else choice) for choice in choices]
        tokenized_example['choices'] = [self.tokenizer(choice, add_special_tokens=False) for choice in choices]
        tokenized_example['gold'] = example['gold']
        return tokenized_example

    def collate_fn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.
        Args:
            data (List): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: dictionary for a single batch
        """
        batch = {
            'input_ids': [],
            'continuation_indices': [],
            'mode': 'icl_task',
            'labels': [],
            'gold_indices': [],
            'choice_groupings': [],
        }
        for data_pair in data:
            choice_start_idx = len(batch['continuation_indices'])

            for choice in data_pair['choices']:
                context_enc = data_pair['preamble']['input_ids'] + data_pair['context']['input_ids']
                continuation_enc = choice['input_ids']
                inp, continuation_span = _make_padded_input(context_enc, continuation_enc, self.max_seq_len,
                                                            self.pad_tok_id)

                batch['input_ids'].append(inp)
                batch['continuation_indices'].append(continuation_span)
                batch['labels'].append(inp)

            batch['gold_indices'].append(data_pair['gold'])
            choice_end_idx = len(batch['continuation_indices'])
            batch['choice_groupings'].append((choice_start_idx, choice_end_idx))

        # We run each distinct query + answer choice through the model separately and determine which
        # answer has the lowest per-token-perplexity.
        #
        # If each question has N possible choices, all N must be grouped together as distinct elements of the batch
        # since the batch may consist of multiple questions, the choice_groupings indicates
        # which contiguous sequences of elements in the batch correspond to which question
        # gold_indices indicates which of the [0, N-1] choices is the correct one for each question.
        batch = {k: torch.stack(v) if k in self.stacked_keys else v for k, v in batch.items()}
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0] // self.num_choices

    def split_batch(self, batch: Any, microbatch_size: int) -> Dict[str, Any]:
        """
        Split batch while ensuring all continuations are in the same microbatch.

        In ICL Multiple Choice, we duplicate each data point for each possible continuation.
        When splitting a batch, we have logical samples, which refer to one possible question,
        and real samples, which refers to one possible continuation. As sample count and
        microbatch_size are tracked in logical samples, we split logical attributes by
        microbatch_size and real attributes by microbatch_size * num_choices.
        Args:
            batch (Dict): batch of data
            microbatch_size (int): size of microbatches

        Returns:
            list: list of chunked batches
        """
        self.check_defaults_are_set({
            'dont_split_keys': self.dont_split_keys,
            'normal_split_keys': self.normal_split_keys
        })
        chunked = {}
        for k, v in batch.items():
            if k in self.dont_split_keys:
                # Defer broadcasting primitives until we know num_chunks
                pass
            elif k == 'continuation_indices':
                # List of lists, so we have to directly call _split_list
                chunked[k] = _split_list(v, microbatch_size * self.num_choices)
            elif k == 'choice_groupings':
                # List of lists, so we have to directly call _split_list
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
    """
    A dataset that constructs batches for in-context learning schema evaluation.
    A schema task involves sentences with a fill-in-the-blank where the user needs to choose the correct word
    to fill in from a set of N options. We use the partial evaluation technique from https://arxiv.org/abs/1806.02847
    to determine the model's choice of fill-in word.

    The default input format is a jsonl file with the following fields:
    - context_options: list of strings corresponding to possible preceding context options for the continuation
    - gold: index of the correct context from 'context_options'
    - continuation: the finishing continuation

    Each batch then consists of batch_size // N distinct tasks and has the following the structure
    - input_ids: Input tensor batch x seqlen x # tokens
    - continuation_indices: List of |batch| consisting of tensors indicating which indices in the sequence correspond to the question answer (aka continuation)
    - mode: Indicates to the model that this is an ICL task and may rely on a custom code path to properly update metrics
    - labels: Identical to the input, used by the model to calculate loss/metrics
    - gold_indices: List of length |batch_size // N| indicating for each question, which of the answers is correct (via an integer [0, N-1])
    - choice_groupings: Indicates which indices of the batch correspond to which questions
    """

    def __init__(self, choices_key='context_options', *args, **kwargs):
        super().__init__(choices_key=choices_key, *args, **kwargs)

    def _construct_context(self, sample, preceding_text: str = '', add_answer: bool = False) -> str:
        """
        Takes a sample and constructs a context with the correct context for the sample's continuation.

        Args:
            sample (Dict): the sample from which to construct the context
            preceding_text (str): any preceding text, needed to if self.example_delimiter is needed at the beginning
            add_answer (bool): this will always be true when calling this function for SchemaTaskDataset

        Returns:
            str: the single correct context for a given continuation

        """
        context_options = sample['context_options']
        gold_idx = sample['gold']
        continuation = sample['continuation']
        context = context_options[gold_idx]
        if len(preceding_text) > 0:
            context = f'{self.example_delimiter}{context}'
        context = f'{context}{self.continuation_delimiter}{continuation}'
        return context

    def _construct_multiple_contexts(self, sample: Dict, preceding_text: str = '') -> str:
        """
        Takes a sample and constructs all contexts. Optionally, appends this to preceeding text (such as a
        prompt or fewshot examples).

        Args:
            sample (Dict): the sample from which to construct the context
            preceding_text (str): any preceding text, needed to if self.example_delimiter is needed at the beginning

        Returns:
            list: all context options for the selected sample with formatting
        """
        context_options = sample['context_options']
        if len(preceding_text) > 0:
            if self.strip_data:
                cont_del = self.continuation_delimiter.rstrip()
            else:
                cont_del = self.continuation_delimiter
            context_options = [f'{self.example_delimiter}{c}{cont_del}' for c in context_options]
        return context_options

    def _prep_example(
        self,
        example: Dict,
        example_idx: int,
        num_fewshot: int,
        prompt_string: str,
        fewshot_rng: random.Random,
    ) -> List[Dict[str, Any]]:
        """
        Prepares a single example from a HF Dataset into tokenized format with prompt and fewshot examples.

        Each task consists of multiple contexts and a single, correct continuation. Will preprend fewshot examples and
        prompt if present.

        Args:
            example (Dict): A dictionary from the hf dataset
            example_idx (int): the index of example
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            fewshot_rng (random.Random): Random number generator to use for fewshot sampling

        Returns:
            Dict: contains a dictionary with the tokenized data
        """
        prompt_and_fewshot = self._generate_few_shot_text(num_fewshot, example_idx, prompt_string, fewshot_rng)
        ctxt = self._construct_multiple_contexts(example, prompt_and_fewshot)
        tokenized_example = self._tokenize_example(prompt_and_fewshot, ctxt, example)
        return tokenized_example

    def _tokenize_example(self, prompt_and_fewshot: str, context_options: List[str], example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handles special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derrived context
            example (Dict): the example as a dictionary.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = {}
        preamble = self.tokenizer(prompt_and_fewshot)
        preamble['input_ids'] = self._fix_eos_on_preamble(preamble['input_ids'])
        tokenized_example['preamble'] = preamble
        tokenized_example['context_options'] = [self.tokenizer(c, add_special_tokens=False) for c in context_options]
        continuation = example['continuation']
        if self.prefix_space:
            continuation = (f' {continuation}' if not continuation.startswith(' ') else continuation)
        tokenized_example['continuation'] = self.tokenizer(continuation, add_special_tokens=False)
        tokenized_example['gold'] = example['gold']
        return tokenized_example

    def collate_fn(self, data) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.
        Args:
            data (List): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: dictionary for a single batch
        """
        batch = {
            'input_ids': [],
            'continuation_indices': [],
            'mode': 'icl_task',
            'labels': [],
            'gold_indices': [],
            'choice_groupings': [],
        }
        for data_pair in data:
            continuation_start_idx = len(batch['continuation_indices'])
            context_options = data_pair['context_options']

            for context in context_options:
                context_enc = data_pair['preamble']['input_ids'] + context['input_ids']
                continuation_enc = data_pair['continuation']['input_ids']
                inp, continuation_span = _make_padded_input(context_enc, continuation_enc, self.max_seq_len,
                                                            self.pad_tok_id)

                batch['input_ids'].append(inp)
                batch['labels'].append(inp)
                batch['continuation_indices'].append(continuation_span)

            batch['gold_indices'].append(data_pair['gold'])
            continuation_end_idx = len(batch['continuation_indices'])
            batch['choice_groupings'].append((continuation_start_idx, continuation_end_idx))

        # We run each distinct query + answer choice through the model separately and determine which
        # answer has the lowest per-token-perplexity.
        #
        # If each question has N possible choices, all N must be grouped together as distinct elements of the batch
        # since the batch may consist of multiple questions, the choice_groupings indicates
        # which contiguous sequences of elements in the batch correspond to which question
        # gold_indices indicates which of the [0, N-1] choices is the correct one for each question.
        batch = {k: torch.stack(v) if k in self.stacked_keys else v for k, v in batch.items()}
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch


class InContextLearningCodeEvalDataset(InContextLearningDataset):
    """
    A dataset that constructs batches for in-context learning code evaluation.

    The default input format is expected to be a jsonl file with the following fields:
    - task_id: label of given task
    - prompt: the code snippet that must be completed
    - entry_point: the entry to the function/code snippet to generate
    - canonical_solution: working solution
    - test: the checker code that will run to completion if the code generation is valid and otherwise throw assertion
    - test_inputs: list of test inputs
    - test_outputs: list of test outputs
    - language: the language of the code snippet

    Each batch then consists of the following the structure
    - input_ids: Input tensor batch x seqlen x num tokens
    - mode: Indicates to the model that this is an ICL task and may rely on a custom code path to properly update metrics
    - mode: always set to 'generate'
    - labels: exact solution for the coding problem
    - prompts: prompt for the task
    - cannonical_solutions: exact solutions
    - entry_points: list of entry points
    - test_inputs: list of test inputs
    - test_outputs: list of test outputs
    - languages:  list of languages
    - pass_at_k: passed value for pass_at_k
    - generation_length: derrived maximum generation length
    - generation_kwargs: Dictionary of kwargs neeeded for generation. Includes the following:
        - pad_token_id: ID for padding token, derived automatically
        - num_beams: how many beams to search for generations, always set to 1
        - num_return_sequences: value passed for 'generations_per_sample', how many generations per prompt
        - do_sample: determines whether model is sampling or greedily decoding. Always set to True
        - top_p: the cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1
        - top_k: the number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity.
        - temperature: randomness used during prediction. 1.0 is deterministic. defaults to 1.0
        - use_cache: Whether or not to use past key values to speed up sampling. Always set to True

    Additional Args:
        # TODO: are these correct?
        generations_per_sample (int) (defaults to 1): how many outputs to generate per prompt
        pass_at_k (int) (defaults to 1): k for how many chances the model gets to write passing code
        top_p (int) (defaults to 0.95): top_p sampling parameter for nucleus sampling
        top_k (int) (defaults to 40): top_k sampling parameter for number of samples to consider
        temperature (float) (defaults to 1.0): temperature to use while sampling
    """

    def __init__(
        self,
        generations_per_sample: int,
        pass_at_k: int = 1,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = 40,
        temperature: Optional[int] = 1.0,
        *args,
        **kwargs,
    ):
        self.check_defaults_are_set({
            'pass_at_k': pass_at_k,
            'generations_per_sample': generations_per_sample,
            'top_p': top_p,
            'top_k': top_k,
            'temperature': temperature
        })
        if generations_per_sample < pass_at_k:
            raise ValueError(
                f'generations_per_sample ({generations_per_sample}) must be greater than or equal to pass_at_k ({pass_at_k}) for code evaluation.'
            )

        super().__init__(
            context_key='prompt',
            answer_key='canonical_solution',
            strip_dataset=False,
            stacked_keys=['input_ids'],
            dont_split_keys=['mode', 'generation_length', 'pass_at_k', 'generation_kwargs'],
            normal_split_keys=['input_ids', 'attention_mask'],
            list_split_keys=[
                'labels', 'tests', 'canonical_solutions', 'entry_points', 'test_inputs', 'test_outputs', 'prompts',
                'languages'
            ],
            *args,
            **kwargs,
        )
        self.pass_at_k = pass_at_k
        self.generations_per_sample = generations_per_sample
        self.max_prompt_length = self.get_max_prompt_length()
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def get_max_prompt_length(self) -> int:
        """
        Iterates through the dataset and finds the length of the longest prompt.
        Returns:
            int: maximum prompt length
        """
        max_prompt_length = 0
        for sample in self.encoded_dataset:
            max_prompt_length = max(
                max_prompt_length,
                len(sample['preamble']['input_ids'] + sample['context']['input_ids']),
            )
        return max_prompt_length

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handles special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derrived context
            example (Dict): the example as a dictionary.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = super()._tokenize_example(prompt_and_fewshot, ctxt, example)
        tokenized_example['prompt_text'] = example['prompt']
        tokenized_example['task_id'] = example['task_id']
        tokenized_example['canonical_solution'] = example['canonical_solution']
        tokenized_example['test'] = example['test']
        tokenized_example['entry_point'] = example['entry_point']
        tokenized_example['test_inputs'] = example['test_inputs']
        tokenized_example['test_outputs'] = example['test_outputs']
        tokenized_example['language'] = example['language']
        return tokenized_example

    def collate_fn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.
        Args:
            data (List): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: dictionary for a single batch
        """
        batch = {
            'input_ids': [],
            'mode': 'generate',
            'labels': [],
            'prompts': [],  # list of prompts
            'tests': [],  # list of tests
            'canonical_solutions': [],  # list of solutions
            'entry_points': [],  # list of entry points
            'test_inputs': [],  # list of test inputs
            'test_outputs': [],  # list of test outputs
            'languages': [],  # list of languages
            'pass_at_k': self.pass_at_k,
            'generation_length': self.max_seq_len - self.max_prompt_length,
            'generation_kwargs': {
                'pad_token_id': self.pad_tok_id,
                # TODO: specify this?
                'num_beams': 1,  # single beam
                'num_return_sequences': self.generations_per_sample,  # how many gens per prompt
                'do_sample': True,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'temperature': self.temperature,
                'use_cache': True
            },
        }
        for sample in data:
            context_enc = sample['preamble']['input_ids'] + sample['context']['input_ids']
            inp, _ = _make_padded_input(
                context_enc,
                [],
                self.max_prompt_length,
                self.pad_tok_id,
                padding_side=self.padding_side,
            )

            batch['input_ids'].append(inp)
            batch['canonical_solutions'].append(sample['canonical_solution'])
            batch['prompts'].append(sample['prompt_text'])
            batch['tests'].append(sample['test'])
            # TODO: why use this twice?
            batch['labels'].append(sample['canonical_solution'])
            batch['entry_points'].append(sample['entry_point'])
            batch['test_inputs'].append(sample['test_inputs'])
            batch['test_outputs'].append(sample['test_outputs'])
            batch['languages'].append(sample['language'])

        batch = {k: torch.stack(v) if k in self.stacked_keys else v for k, v in batch.items()}
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
    hf_loading_vars: Dict,
    hf_parsing_map: Dict,
    destination_path: str,
    prelimiter: str,  # e.g. 'Question: '
    cot_delimiter: str,
    fewshot_random_seed: int,
    pass_at_k: int,
    generations_per_sample: int,
    temperature: float,
) -> DataSpec:
    if icl_task_type == 'multiple_choice':
        dataset = InContextLearningMultipleChoiceTaskDataset(
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_tok_id=pad_tok_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            continuation_delimiter=continuation_delimiter,
            destination_path=destination_path,
            fewshot_random_seed=fewshot_random_seed,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
        )
        batch_size = max(dataset.num_choices, batch_size)
        effective_batchsize = batch_size // dataset.num_choices
    elif icl_task_type == 'schema':
        dataset = InContextLearningSchemaTaskDataset(
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_tok_id=pad_tok_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            continuation_delimiter=continuation_delimiter,
            destination_path=destination_path,
            fewshot_random_seed=fewshot_random_seed,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
        )
        batch_size = max(dataset.num_choices, batch_size)
        effective_batchsize = batch_size // dataset.num_choices
    elif icl_task_type == 'language_modeling':
        dataset = InContextLearningLMTaskDataset(
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_tok_id=pad_tok_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            continuation_delimiter=continuation_delimiter,
            destination_path=destination_path,
            fewshot_random_seed=fewshot_random_seed,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
        )
        effective_batchsize = batch_size
    elif icl_task_type == 'question_answering':
        dataset = InContextLearningQATaskDataset(
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_tok_id=pad_tok_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            continuation_delimiter=continuation_delimiter,
            destination_path=destination_path,
            prelimiter=prelimiter,
            fewshot_random_seed=fewshot_random_seed,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
            cot_delimiter=cot_delimiter,
        )
        effective_batchsize = batch_size
    elif icl_task_type == 'code_evaluation':
        dataset = InContextLearningCodeEvalDataset(
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_tok_id=pad_tok_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            continuation_delimiter=continuation_delimiter,
            destination_path=destination_path,
            prelimiter=prelimiter,
            fewshot_random_seed=fewshot_random_seed,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
            pass_at_k=pass_at_k,
            generations_per_sample=generations_per_sample,
            temperature=temperature,
        )
        effective_batchsize = batch_size
    elif icl_task_type == 'rag':
        dataset = InContextLearningRAGGenerationTaskDataset(
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_tok_id=pad_tok_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            continuation_delimiter=continuation_delimiter,
            passage_delimiter='\nPassage: ',
            passage_query_delimiter='\nQuery: ',
            destination_path=destination_path,
            fewshot_random_seed=fewshot_random_seed,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,   
        )
        effective_batchsize = batch_size
    else:
        raise Exception(f'Unrecognized ICL task type: {icl_task_type}')

    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)

    split_batch = None
    if isinstance(
            dataset,
        (
            InContextLearningMultipleChoiceTaskDataset,
            InContextLearningQATaskDataset,
            InContextLearningCodeEvalDataset,
            InContextLearningRAGGenerationTaskDataset,
        ),
    ):
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


def partition_dataset_by_category(dataset_uri: str, destination_path: str, hf_loading_vars: Dict,
                                  hf_parsing_map: Dict) -> Dict[str, str]:
    """
    If has_categories is enabled, we partition the dataset into a separate dataset for each category value in the data and write each partition to a local file.

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
        raise MissingConditionalImportError(
            extra_deps_group='nlp',
            conda_package='datasets',
            conda_channel='conda-forge',
        ) from e
    if _check_if_huggingface_uri(dataset_uri):
        dataset = load_dataset(dataset_uri, **hf_loading_vars)
        if hf_parsing_map:
            dataset_parsing_func = lambda example: {
                k: ' '.join([str(example[col]) for col in v]) for k, v in hf_parsing_map.items()
            }
            dataset = dataset.map(dataset_parsing_func, remove_columns=dataset.column_names)
    else:
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
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
    max_seq_len: int,
    pad_tok_id: int,
    num_fewshot: int,
    prompt_string: str,  # e.g. 'translate english to french:'
    example_delimiter: str,  # e.g. '\n'
    continuation_delimiter: str = '',
    question_prelimiter: str = '',  # e.g. 'Question: '
    hf_loading_vars: Dict = None,
    hf_parsing_map: Dict = None,
    destination_path: str = '',
    fewshot_random_seed: int = 1234,
    pass_at_k: int = 1,
    temperature: float = 1.0,
    generations_per_sample: int = 1,
    cot_delimiter: str = '',
    has_categories: bool = False,
) -> Union[DataSpec, Dict[str, DataSpec]]:
    """
    This constructs a dataloader (or dataloaders if has_categories is True) capable of evaluating LLMs on in-context learning language modeling tasks, for example LAMBADA. An example usage is below:

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
        icl_task_type (str): Name of icl_task type. One of ['multiple_choice', 'schema', 'language_modeling', 'question_answering', 'code_evaluation']
        dataset_uri (str): Either a local path, a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`, a link to a HuggingFace Dataset
        tokenizer (transformers.PreTrainedTokenizerBase): The tokenizer used to transform data into batches
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The sequence length expected by the model
        pad_tok_id (int): The special token reserved for padding the ends of batches
        num_fewshot (int): The number of complete fewshot examples to pad each test example with
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'translate english to french')
        example_delimiter (str): Separator that goes between individual examples (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        question_prelimiter: (str): Text to be prepended before each context segement in each eval example. (e.g. 'Q:', 'The following is a paragraph containing...')
        hf_loading_vars (Dict): A dictionary containing keyword arguments to be passed into `load_dataset` if dataset is being pulled from HF.
        hf_parsing_map (Dict[str:List[str]]): A dictionary containing a from HF columns to ICL dataset keys. The dictionary should be formatted {icl_key:[hf_key1, hf_key1]}.
            Values in the dict will be concatenated with ' ' seperating them. If not included, will use the columns already present in the HF dataset.
        destination_path: (str): This is the local file where remote datasets will be saved.
        fewshot_random_seed (int): Random seed to use for fewshot sampling
        # TODO: is this right?
        pass_at_k (int): k for how many chances the model gets to write passing code
        generations_per_sample (int): how many outputs to generate per prompt
        cot_delimiter (str): Delimiter to place between the chain of thought and continuations.
        has_categories: (bool): If ``True``, we will search the dataset file for a category key, and partition the dataset into a separate dataloader for each category occurring in the data.

    Returns:
        DataLoader: A dataloader used for performing in-context learning evaluation on the dataset provided.
    """

    if has_categories:
        result_dls = {}
        output_files = partition_dataset_by_category(dataset_uri, destination_path, hf_loading_vars, hf_parsing_map)
        categories = sorted(output_files.keys())
        for category in categories:
            partition_uri = output_files[category]
            result_dls[category] = build_icl_dataloader(
                icl_task_type=icl_task_type,
                dataset_uri=partition_uri,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pad_tok_id=pad_tok_id,
                num_fewshot=num_fewshot,
                prompt_string=prompt_string,
                example_delimiter=example_delimiter,
                continuation_delimiter=continuation_delimiter,
                destination_path=partition_uri + '_tmp',
                prelimiter=question_prelimiter,
                cot_delimiter=cot_delimiter,
                fewshot_random_seed=fewshot_random_seed,
                pass_at_k=pass_at_k,
                generations_per_sample=generations_per_sample,
                hf_loading_vars=hf_loading_vars,
                hf_parsing_map=hf_parsing_map,
                temperature=temperature,
            )
        return result_dls
    else:
        return build_icl_dataloader(
            icl_task_type=icl_task_type,
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            pad_tok_id=pad_tok_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
            continuation_delimiter=continuation_delimiter,
            destination_path=destination_path,
            prelimiter=question_prelimiter,
            cot_delimiter=cot_delimiter,
            fewshot_random_seed=fewshot_random_seed,
            pass_at_k=pass_at_k,
            generations_per_sample=generations_per_sample,
            temperature=temperature,
        )
