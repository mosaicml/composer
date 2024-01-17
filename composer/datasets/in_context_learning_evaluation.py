# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

import copy
import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

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


def strip_data(example: Dict) -> Dict:
    """
    Remove white space from the begging and end of string values in a dictionary

    Args:
        example: dictionary to be stripped

    Returns:
        dict: the same dictionary with .strip() applied to any value in the dict that is a string
    """
    return {k: v.strip() if isinstance(v, str) else v for k, v in example.items()}


def _tokenizer_needs_prefix_space(tokenizer: transformers.PreTrainedTokenizerBase) -> bool:
    """
    Test for whether a prefix space is needed before the continuation.
    Sentencepiece tokenization should not have a prefix space, but gpt2 style BPE should.

    Args:
        tokenizer: Tokenizer to test

    Returns:
        bool: whether or not the tokenizer needs a prefix space
    """
    return len(tokenizer(' a', add_special_tokens=False)['input_ids']) == 1


def _trim_context(context_enc: List, continuation_enc: List, max_seq_len: int) -> List:
    """
    Trims a list of tokens down to `max_seq_len` if the length of the list plus the continuation
    is more than `max_seq_len`. It will always trim tokens from the left, i.e. tokens at the beginning
    of the context will be removed. 
    
    Args:
        context_enc (list): list of tokens in the context
        continuation_enc (lsit): list of tokens in the continuation
        max_seq_len (int): maximum length the model can ingest

    Returns:
        list: the encoded context trimmed from the left
    """
    if len(continuation_enc) + len(context_enc) > max_seq_len:
        context_max_subseq_len = max_seq_len - len(continuation_enc)

        if context_max_subseq_len < 0:
            # can't support continuations which are longer than the max seq len
            raise Exception(f'Dataset included continuation longer than the max seq len')

        # clip from the end
        context_enc = context_enc[-(context_max_subseq_len):]
    return context_enc


def _get_continuation_span(context_enc: List, continuation_enc: List) -> list:
    """
    Gets the list of indices of the continutaion tokens for language modeling or generaiton tasks.

    Args:
        context_enc (list): list of context tokens
        continuation_enc (list): list of continuation tokens

    Returns:
        torch.tensor: a tensor containing indices corresponding to continuation tokens 
    """
    return torch.tensor(range(len(context_enc), len(context_enc) + len(continuation_enc)))


def _make_padded_input(context_enc: List,
                       continuation_enc: List,
                       max_seq_len: int,
                       pad_tok_id: int,
                       padding_side: str = 'right') -> Tuple[torch.tensor, torch.tensor]:
    """
    Takes an encoded context and continuation and clips the beginning of the context if they're too long.
    Adds the padding token to the specified side.

    Args:
        context_enc (List): the encoded input to the model
        continuation_enc (List): the encoded desired output for the example
        max_seq_list (int): maximum length sequences can be
        pad_tok_id (int): the token id we pad with
        padding_side (str): which side to pad the context on. Can be 'right' or 'left

    Returns:
        input (torch.tensor): the padded and encoded context
        continuation_span (torch.tensor): the _inclusive_ range of indices corresponding to the continuation
    """

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

    return inp


def convert_tokens_to_tensors(batch: Dict, tokenize_labels: bool) -> Dict[str, Any]:
    """
    HF Datasets converts tensors into lists when we store them, and we don't want to use `type='torch'`
    because some content in the dataset, like generation args or single ints, should not be converted.

    Here, we convert those lists of tokens back into tensors in order to feed them into the model.

    Args:
        batch (dict): a dictionary of batched inputs
        tokenize_labels (bool): whether or not the labels are tokenized (and need to be stacked)

    Returns:
        dict: the batch with torch tensors in the corresponding keys instead of lists of lists
    """
    batch['input_ids'] = torch.stack(list(map(torch.tensor, batch['input_ids'])))
    if tokenize_labels:
        batch['labels'] = torch.stack(list(map(torch.tensor, batch['labels'])))
        batch['continuation_indices'] = list(map(torch.tensor, batch['continuation_indices']))
    return batch


def _get_fewshot_sample_idxs(dataset_size: int, num_fewshot: int, example_idx: int, rng: random.Random) -> List[int]:
    """
    Samples indices without replacement. If num_fewshot exceeds the number of unique examples in the dataset,
    then we will have fewer than num_fewshot examples in context.
    Args:
        dataset_size (int): length of the dataset
        num_fewshot (int): number of examples to prepend
        example_idx (int): current example's index (excluded from fewshot choices)
        rng (random.Random): rng for repeatable sample selection

    Returns:
        list: indices of the examples chosen for fewshot selection
    """
    num_fewshot = min(dataset_size - 1, num_fewshot)
    fewshot_idxs = set(rng.sample(range(0, dataset_size), num_fewshot))

    if example_idx in fewshot_idxs:
        fewshot_idxs.remove(example_idx)
        if len(fewshot_idxs) >= dataset_size - 1:
            return fewshot_idxs

        replacement_sample = rng.choice(range(0, dataset_size))
        while replacement_sample in fewshot_idxs or replacement_sample == example_idx:
            replacement_sample = rng.choice(range(0, dataset_size))
        fewshot_idxs.add(replacement_sample)
    return fewshot_idxs


class InContextLearningDataset(Dataset):
    """
    A base dataset that constructs batches for in-context learning task evaluations.
    The dataset format is expected to be a local jsonl file, a cloud link to a jsonl file, or a Hugging Face dataset link.
    'context' refers to the input a model will recieve before generating an output. For example, the question in question answering tasks,
        the preceding text in a language modeling task, or the document and question regarding the document in a document understanding task.
    'example' refers to an loaded dictionary, generally containing a context, an answer, and any other information needed to run the task.
    'answer' refers to the desired output of the model.

    When creating a new ICL Dataset, it is likely that you will need to reimplemente the following methods:
        - _construct_context(): takes a single example dictionary and formulates the context as a string for that eval question.
        - _get_answer_from_example(): takes a single example dictionary and formulates the correct, ground truth answer as a string.
        - _tokenize_example(): tokenizes the example and adds any extra content from the original dictionary that needs to be passed downstream.
        - _read_dataset(): loads the dataset and does basic parsing. If additional parsing must be done, this is a good place to do so (See InContextLearningQATaskDataset._read_dataset())

    Additionally, base_batch and batch_mapping must be defined.
        - base_batch (Dict): the base dictionary that the dataset will use to construct a batch. This should contain static values, like generation_kwargs or mode,
                             and empty lists for values that will need to be accumulated from each example.
                             NOTE: Sometimes you will need to set base_batch directly after the init call, e.g. in order to use class variables
                                   like self.pad_tok_id or self.max_answer_length. If you manually set generation_kwargs this way, you'll need to call self._update_generation_kwargs()
                                   after setting self.base_batch.
        - batch_mapping (Dict): A mapping with keys that are keys in the batch and values that are columns in the loaded dataset.
                                collate_fn will use this mapping to create batches from self.dataset.

    Args:
        dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri prepended with ``hf://``.
            Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            A local dataset must consist of rows of JSON data points with task dependant fields.
            The default keys expected are "context" and "answer".
        tokenizer (transformers.PreTrainedTokenizerBase): The tokenizer used to map between strings and token ids.
        max_seq_len (int): The maximum sequence length supported by the model.
        pad_tok_id (int): The special token used for padding batches.
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example. These are not identical across examples.
        fewshot_random_seed (int): Random seed to use for fewshot sampling.
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'Translate english to french.').
        example_delimiter (str): Separator inserted before (context, answer) pairs (e.g. '\n') for fewshot sampling and prompting.
        continuation_delimiter: (str): Separator inserted between context and answer in each example (e.g. '\nA: ').
        destination_path (str): Temporary path to store downloaded datasets.
        prelimiter (str): Text to be prepended before each context, including few shot examples (e.g. "Question: ").
        context_key (str): The key in the loaded dataset that contains the context.
        answer_key (str): The key in the loaded dataset that contains the answer.
        strip_dataset (bool): Boolean for whether to strip whitespace from data. Trailing whitespace can cause degenerative outputs,
            so unless whitespace should be preserved (for example in code), this should be set to True.
        padding_side (str): Side of the content and answer on which to apply padding. Can be either 'right' or 'left'.
        padding_size (int): The final size of the tensor after padding. Defaults to max_sequence_length.
        base_batch (Dict): The base dictionary upon which a batch is created. See above for more details.
        base_mapping (Dict): A mapping of batch keys to dataset columns, used to create batches. See above for more details.
        hf_loading_vars (Dict): A dictionary containing keyword arguments to be passed into `load_dataset` if dataset is being pulled from HF.
        hf_parsing_map (Dict): A dictionary containing a mapping from HF columns to ICL dataset keys. The dictionary should be formatted {icl_key:[hf_key1, hf_key1]}.
            Column contents will be concatenated with ' ' seperating them. If not included, will load the columns already present in the HF dataset.
        tokenize_labels (bool): Whether or not the labels should be tokenized. Generally determined by which metric a dataset uses.
        generation_kwargs (Dict): A dictionary containing keyword arguments to be passed along to the model's generate function.
    """

    def __init__(
        self,
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
        static_keys: List = None,
        list_keys: List = None,
        tensor_keys: List = None,
        prelimiter: str = '',
        context_key: str = 'context',
        answer_key: str = 'answer',
        strip_dataset: bool = True,
        padding_side: str = 'right',
        padding_size: int = None,
        base_batch: Dict = None,
        batch_mapping: Dict = None,
        hf_loading_vars: Dict = None,
        hf_parsing_map: Dict = None,
        tokenize_labels: bool = True,
        generation_kwargs: Dict = None,
    ):
        self.tokenizer = tokenizer
        self.prefix_space = _tokenizer_needs_prefix_space(self.tokenizer)

        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.num_fewshot = num_fewshot
        self.padding_side = padding_side
        self.padding_size = padding_size if padding_size else self.max_seq_len
        self.prelimiter = prelimiter
        self.example_delimiter = example_delimiter
        self.continuation_delimiter = continuation_delimiter
        self.context_key = context_key
        self.answer_key = answer_key
        self.tokenize_labels = tokenize_labels
        self.batch_mapping = batch_mapping or {}
        self.base_batch = base_batch or {}
        self._update_generation_kwargs(generation_kwargs or {})

        self.static_keys = static_keys
        self.list_keys = list_keys
        self.tensor_keys = tensor_keys

        hf_loading_vars = hf_loading_vars or {}
        self.dataset = self._read_dataset(dataset_uri, destination_path, hf_loading_vars, hf_parsing_map)
        self.strip_data = strip_dataset
        if self.strip_data:
            self.dataset = self.dataset.map(strip_data)

        fewshot_rng = random.Random(fewshot_random_seed)
        self.dataset = self.dataset.map(
            self._prep_example,
            with_indices=True,
            fn_kwargs={
                'num_fewshot': num_fewshot,
                'prompt_string': prompt_string,
                'fewshot_rng': fewshot_rng,
            },
        )

    def __getitem__(self, index: int) -> Dict:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def get_num_samples_in_batch(self, batch: Dict) -> int:
        return batch['input_ids'].shape[0]

    def _update_generation_kwargs(self, generation_kwargs: Dict) -> None:
        """
        Updates self.base_batch with the passed in generation_kwargs.
        This must be run after self.base_batch is set (for example, if self.base_batch is set after __init__() is run,
        likely because base_batch needs a class variable like self.pad_tok_id or self.max_answer_length).

        Args:
            dict: keyword arguments that be written into base_batch['generation_kwargs']
        """
        if 'generation_kwargs' not in self.base_batch:
            self.base_batch['generation_kwargs'] = {}
        if generation_kwargs:
            self.base_batch['generation_kwargs'].update(generation_kwargs)

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
        if 'hf://' in dataset_uri:
            dataset_uri = dataset_uri.replace('hf://', '')
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

    def _generate_few_shot_prompt(
        self,
        num_fewshot: int,
        example_idx: int,
        preamble: str,
        fewshot_rng: random.Random,
    ) -> str:
        """
        Formats the fewshot prompt for test example `example_idx`.

        Randomly selects `num_fewshot` samples from the dataset (excluding the example at `example_idx`) and constructs
        contextes with answers appended.

        Returns the formatted prompt_string + concatenated list of formatted few shot examples as a string.

        Args:
            num_fewshot (int): number of examples to prepend
            example_idx (int): current example idx
            preamble (str): text to occur at the beginning of the task. Generally instructions or a prompt.
            fewshot_rng (random.Random): seeded sampler to chose samples with

        Returns:
            str: the original preamble with num_fewshot examples appended
        """
        few_shot_text = preamble

        if num_fewshot > 0:
            fewshot_idxs = _get_fewshot_sample_idxs(len(self.dataset), num_fewshot, example_idx, fewshot_rng)
            for fewshot_idx in fewshot_idxs:
                ctxt = self._construct_context(self.dataset[fewshot_idx], few_shot_text, add_answer=True)
                few_shot_text += ctxt

        return few_shot_text

    def _construct_context(self, example: Dict, preceding_text: str = '', add_answer: bool = False) -> str:
        """
        Takes an example and constructs a context, i.e. the input the model reads for this example.
        Optionally adds the correct answer (for fewshot examples) and handles example delimiters

        Args:
            example (Dict): the example from which to construct the context
            preceding_text (str): any preceding text, used as a check for prepending self.example_delimiter
            add_answer (bool): bool for whether or not to add the answer on the end of the context (e.g. for fewshot examples)

        Returns:
            str: The constructed context. The default output context is
                 formatted as follows: f'{self.prelimiter}{example[self.context_key]}{self.continuation_delimiter}'
        """
        ctxt = example[self.context_key]
        ctxt = f'{self.prelimiter}{ctxt}'
        if len(preceding_text) > 0:
            ctxt = f'{self.example_delimiter}{ctxt}'
        ctxt = f'{ctxt}{self.continuation_delimiter}'
        if add_answer:
            ctxt = f'{ctxt}{self._get_answer_from_example(example, in_context=add_answer)}'
        return ctxt

    def _get_answer_from_example(self, example: Dict[str, Any], in_context: bool = False) -> str:
        """
        Returns the answer from the example.

        Args:
            example (Dict): the example from which to retrieve the answer

        Returns:
            str: the answer in the example
        """
        cont = example[self.answer_key]
        if self.prefix_space and not cont.startswith(' ') and not in_context:
            cont = f' {cont}'
        return cont

    def _fix_eos_on_preamble(self, input_ids: List[int]) -> List[int]:
        """
        If the input_ids is empty then input_ids will be a 0-length List
        unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer).
        If there is an EOS token added, we need to remove it so it is not in the middle of the prompt,
        as the specific eval question's prompt will follow the input_ids.

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
        Runs text through the tokenizer and handle special cases.

        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctxt (str): the specific example's derrived context
            example (Dict): the example as a dictionary. Used for additional processing in inherited classes.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = {}
        # Always add special tokens to preamble
        preamble = self.tokenizer(prompt_and_fewshot)
        preamble = self._fix_eos_on_preamble(preamble['input_ids'])
        if self.strip_data:
            # rstrip context because a prompt ending in a space results in degenerate output
            ctxt = ctxt.rstrip()
        # Never add special tokens to context
        tokenized_context = self.tokenizer(ctxt, add_special_tokens=False)['input_ids']
        tokenized_context = preamble + tokenized_context

        if self.tokenize_labels:
            # Never add special tokens to answer
            tokenized_answer = self.tokenizer(self._get_answer_from_example(example),
                                              add_special_tokens=False)['input_ids']
            trimmed_context = _trim_context(tokenized_context, tokenized_answer, self.padding_size)
            continuation_indices = _get_continuation_span(trimmed_context, tokenized_answer)
            padded_context = _make_padded_input(trimmed_context, tokenized_answer, self.padding_size, self.pad_tok_id,
                                                self.padding_side)

            tokenized_example[self.context_key] = padded_context
            tokenized_example[self.answer_key] = tokenized_answer
            tokenized_example['continuation_indices'] = continuation_indices
        else:
            trimmed_context = _trim_context(tokenized_context, [], self.padding_size)
            padded_context = _make_padded_input(trimmed_context, [], self.padding_size, self.pad_tok_id,
                                                self.padding_side)

            tokenized_example[self.context_key] = padded_context
            tokenized_example[self.answer_key] = self._get_answer_from_example(example)

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
        prompt_and_fewshot = self._generate_few_shot_prompt(num_fewshot, example_idx, prompt_string, fewshot_rng)
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
        batch = copy.deepcopy(self.base_batch)
        for data_pair in data:
            for batch_key, data_key in self.batch_mapping.items():
                batch[batch_key].append(data_pair[data_key])
            if 'continuation_indices' in data_pair:
                batch['continuation_indices'].append(data_pair['continuation_indices'])

        batch = convert_tokens_to_tensors(batch, self.tokenize_labels)
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
        chunked = {}
        for k, v in batch.items():
            if k in self.static_keys:
                # Defer broadcasting until we know num_chunks
                pass
            elif k in self.list_keys:
                chunked[k] = _split_list(v, microbatch_size)
            elif k in self.tensor_keys:
                chunked[k] = _default_split_batch(v, microbatch_size)
            else:
                raise ValueError(f'Unexpected key {k} in batch splitting')
        num_chunks = len(chunked['input_ids'])
        for k, v in batch.items():
            if k in self.static_keys:
                chunked[k] = [v] * num_chunks

        batched_list = [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]
        return batched_list


class InContextLearningQATaskDataset(InContextLearningDataset):
    """
    A dataset that constructs batches for in-context learning question answering evaluation.
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
        self.max_answer_length = 0
        static_keys = ['mode', 'cot_delimiter', 'generation_length', 'generation_kwargs']
        tensor_keys = ['input_ids', 'attention_mask']
        list_keys = ['labels']
        super().__init__(padding_side='left',
                         tokenize_labels=False,
                         static_keys=static_keys,
                         list_keys=list_keys,
                         tensor_keys=tensor_keys,
                         *args,
                         **kwargs)
        # NOTE: set these after init call bcus they take class vars
        self.base_batch = {
            'input_ids': [],
            'mode': 'generate',
            'labels': [],
            'cot_delimiter': self.cot_delimiter,
            'generation_length': self.max_answer_length,
            'generation_kwargs': {
                'pad_token_id': self.pad_tok_id,
                'use_cache': True
            }
        }
        self.batch_mapping = {
            'input_ids': self.context_key,
            'labels': 'aliases',
        }
        self._update_generation_kwargs(kwargs.get('generation_kwargs'))

    def _read_dataset(
        self,
        dataset_uri: str,
        destination_path: str,
        hf_loading_vars: Dict = None,
        hf_parsing_map: Dict = None,
    ):
        dataset = super()._read_dataset(dataset_uri, destination_path, hf_loading_vars, hf_parsing_map)
        self.has_cot = 'chain_of_thought' in dataset.features
        dataset = dataset.map(
            lambda examples: {
                'context': examples['context'],
                'answer': examples['answer'],
                'aliases': set([examples['answer']] + examples.get('aliases', [])),
                'chain_of_thought': examples.get('chain_of_thought', ''),
            })
        self.max_answer_length = self._get_max_answer_length(dataset)
        # NOTE: This is the only time we use the class variable padding_size.
        self.padding_size = self.max_seq_len - self.max_answer_length
        return dataset

    def _get_answer_from_example(self, example: Dict, in_context=False) -> str:
        """
        Returns the answer from the example. Applies chain of thought if self.has_cot is marked as true.
        Args:
            example (Dict): the example from which to retrieve the answer

        Returns:
            str: the answer in from the example with chain of thought and delimiter if needed
        """
        if self.has_cot:
            return f'{example["chain_of_thought"]}{self.cot_delimiter}{example[self.answer_key]}'
        else:
            return example[self.answer_key]

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handle special cases.
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

    def _get_max_answer_length(self, dataset) -> int:
        f"""
        Loops over the dataset and finds the longest answer length.

        Returns:
            int: the maximum answer length with an additional buffer of {_MAX_ANSWER_BUFFER_LENGTH} if chain of thought is present
        """
        max_answer_length = 0
        for example in dataset:
            all_answers = [example[self.answer_key]] + list(example.get('aliases', []))
            for answer in all_answers:
                if self.has_cot:
                    response = (f'{example["chain_of_thought"]}{self.cot_delimiter}{answer}')
                else:
                    response = answer
                max_answer_length = max(max_answer_length, len(self.tokenizer(response)['input_ids']))
        max_answer_length = max_answer_length + (_MAX_ANSWER_BUFFER_LENGTH if len(self.cot_delimiter) > 0 else 0)
        return max_answer_length


class InContextLearningLMTaskDataset(InContextLearningDataset):
    """
    A dataset that constructs batches for in-context learning language modeling evaluation.
    Language modeling tasks test a model's ability to properly predict tokens based on preceding tokens.

    The input format is expected to be a jsonl file with the following fields:
    - context: preceding text
    - continuation: the expected continuation

    See InContextLearningDataset for more details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(answer_key='continuation',
                         static_keys=['mode'],
                         tensor_keys=['input_ids', 'continuation_indices', 'labels', 'attention_mask'],
                         base_batch={
                             'input_ids': [],
                             'continuation_indices': [],
                             'mode': 'icl_task',
                             'labels': []
                         },
                         batch_mapping={
                             'input_ids': 'context',
                             'labels': 'context'
                         },
                         padding_side='right',
                         *args,
                         **kwargs)


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

    def __init__(self,
                 choices_key: str = 'choices',
                 static_keys: List = None,
                 list_of_tensors_keys: List = None,
                 list_of_tuples_keys: List = None,
                 list_of_primitives: List = None,
                 *args,
                 **kwargs):
        self.choices_key = choices_key
        base_batch = {
            'input_ids': [],
            'continuation_indices': [],
            'mode': 'icl_task',
            'labels': [],
            'gold_indices': [],
            'choice_groupings': [],
        }
        context_key = kwargs.pop('context_key', 'query')
        static_keys = kwargs.pop('static_keys', ['mode', 'generation_kwargs'])
        tensor_keys = kwargs.pop('tensor_keys', ['input_ids', 'labels', 'attention_mask'])
        self.list_of_tensors_keys = list_of_tensors_keys or ['continuation_indices']
        self.list_of_tuples_keys = list_of_tuples_keys or ['choice_groupings']
        self.list_of_primitives = list_of_primitives or ['gold_indices']
        super().__init__(context_key=context_key,
                         base_batch=base_batch,
                         static_keys=static_keys,
                         tensor_keys=tensor_keys,
                         padding_side='right',
                         *args,
                         **kwargs)
        self.num_choices = len(self.dataset[0][self.choices_key])
        self.batch_mapping_per_choice = {'input_ids': 'context', 'labels': 'context'}
        self.batch_map_per_example = {'gold_indices': 'gold'}

    def _get_answer_from_example(self, example: Dict, in_context=False) -> str:
        """
        Returns the correct answer from the example's choices.
        Args:
            example (Dict): the example from which to retrieve the answer

        Returns:
            str: the full string of the correct answer based on the 'gold' key
        """
        choices = example[self.choices_key]
        gold_idx = example['gold']
        return choices[gold_idx]

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handle special cases.
        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derrived context
            example (Dict): the example as a dictionary.

        Returns:
            Dict: dictionary with the tokenized data
        """
        # NOTE: some of this is repeated from super class but for loop makes things considerably different
        tokenized_example = {}
        # Always add special tokens to preamble
        preamble = self.tokenizer(prompt_and_fewshot)
        preamble = self._fix_eos_on_preamble(preamble['input_ids'])
        if self.strip_data:
            # rstrip context because a prompt ending in a space results in degenerate output
            ctxt = ctxt.rstrip()
        # Never add special tokens to context
        tokenized_context = self.tokenizer(ctxt, add_special_tokens=False)['input_ids']
        tokenized_context = preamble + tokenized_context

        tokenized_example[self.context_key] = []
        tokenized_example[self.answer_key] = []
        tokenized_example['continuation_indices'] = []
        # NOTE: Treating tokenize_labels as True for all MC datasets (required for our MC accuracy metric)
        for choice in example[self.choices_key]:
            if self.prefix_space:
                choice = f' {choice}' if not choice.startswith(' ') else choice

            # Never add special tokens to answer
            tokenized_answer = self.tokenizer(choice, add_special_tokens=False)['input_ids']
            trimmed_context = _trim_context(tokenized_context, tokenized_answer, self.padding_size)
            continuation_indices = _get_continuation_span(trimmed_context, tokenized_answer)
            padded_context = _make_padded_input(trimmed_context, tokenized_answer, self.padding_size, self.pad_tok_id,
                                                self.padding_side)

            tokenized_example[self.context_key].append(padded_context)
            tokenized_example[self.answer_key].append(tokenized_answer)
            tokenized_example['continuation_indices'].append(continuation_indices)

        tokenized_example['gold'] = example['gold']
        return tokenized_example

    def collate_fn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.
        We run each distinct query + answer choice through the model separately and determine which
        answer has the lowest per-token-perplexity.

        If each question has N possible choices, all N must be grouped together as distinct elements of the batch
        since the batch may consist of multiple questions, the choice_groupings indicates
        which contiguous sequences of elements in the batch correspond to which question
        gold_indices indicates which of the [0, N-1] choices is the correct one for each question.
        Args:
            data (List): list of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: dictionary for a single batch
        """
        batch = copy.deepcopy(self.base_batch)
        for data_pair in data:
            choice_start_idx = len(batch['continuation_indices'])
            # TODO: use batch_mappings? Could be fine as is
            for i, context_enc in enumerate(data_pair[self.context_key]):
                batch['input_ids'].append(context_enc)
                batch['continuation_indices'].append(data_pair['continuation_indices'][i])
                batch['labels'].append(context_enc)

            batch['gold_indices'].append(data_pair['gold'])
            choice_end_idx = len(batch['continuation_indices'])
            batch['choice_groupings'].append((choice_start_idx, choice_end_idx))

        batch = convert_tokens_to_tensors(batch, self.tokenize_labels)
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0] // self.num_choices

    def split_batch(self, batch: Any, microbatch_size: int) -> Dict[str, Any]:
        """
        Split batch while ensuring all continuations are in the same microbatch.

        In ICL Multiple Choice, we duplicate each data point for each possible continuation.
        When splitting a batch, we have logical example, which refer to one possible question,
        and real example, which refers to one possible continuation. As example count and
        microbatch_size are tracked in logical example, we split logical attributes by
        microbatch_size and real attributes by microbatch_size * num_choices.
        Args:
            batch (Dict): batch of data
            microbatch_size (int): size of microbatches

        Returns:
            list: list of chunked batches
        """
        chunked = {}
        for k, v in batch.items():
            if k in self.static_keys:
                # Defer broadcasting primitives until we know num_chunks
                pass
            elif type(v) == list:
                # list of tensors - 'continuation_indices'
                if k in self.list_of_tensors_keys:
                    chunked[k] = _split_list(v, microbatch_size * self.num_choices)
                # list of tuples - 'choice_groupings'
                elif k in self.list_of_tuples_keys:
                    chunked[k] = _split_list(v, microbatch_size)
                # list - 'gold_indices'
                elif k in self.list_of_primitives:
                    chunked[k] = _default_split_batch(v, microbatch_size)
                else:
                    raise ValueError(f'Unexpected key {k} in list splitting')
            elif k in self.tensor_keys:
                chunked[k] = _default_split_batch(v, microbatch_size * self.num_choices)
            else:
                raise ValueError(f'Unexpected key {k} in batch splitting')
        num_chunks = len(chunked['input_ids'])
        # Broadcast primitives to all chunks
        for k, v in batch.items():
            if k in self.static_keys:
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
        static_keys = ['mode']
        tensor_keys = ['input_ids', 'labels', 'attention_mask']
        list_of_tensors_keys = ['continuation_indices']
        super().__init__(choices_key=choices_key,
                         context_key=choices_key,
                         static_keys=static_keys,
                         tensor_keys=tensor_keys,
                         list_of_tensors_keys=list_of_tensors_keys,
                         *args,
                         **kwargs)
        self.base_batch = {
            'input_ids': [],
            'continuation_indices': [],
            'mode': 'icl_task',
            'labels': [],
            'gold_indices': [],
            'choice_groupings': [],
        }

    def _construct_context(self, example, preceding_text: str = '', add_answer: bool = False) -> str:
        """
        Takes a example and constructs a context with the correct context for the example's continuation.

        Args:
            example (Dict): the example from which to construct the context
            preceding_text (str): any preceding text, needed to if self.example_delimiter is needed at the beginning
            add_answer (bool): this will always be true when calling this function for SchemaTaskDataset

        Returns:
            str: the single correct context for a given continuation
        """
        context_options = example[self.choices_key]
        gold_idx = example['gold']
        continuation = example['continuation']
        context = context_options[gold_idx]
        if len(preceding_text) > 0:
            context = f'{self.example_delimiter}{context}'
        context = f'{context}{self.continuation_delimiter}{continuation}'
        return context

    def _construct_multiple_contexts(self, example: Dict, preceding_text: str = '') -> str:
        """
        Takes a example and constructs all contexts. Optionally, appends this to preceeding text (such as a
        prompt or fewshot examples).

        Args:
            example (Dict): the example from which to construct the context
            preceding_text (str): any preceding text, needed to if self.example_delimiter is needed at the beginning

        Returns:
            list: all context options for the selected example with formatting
        """
        context_options = example[self.choices_key]
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
        prompt_and_fewshot = self._generate_few_shot_prompt(num_fewshot, example_idx, prompt_string, fewshot_rng)
        ctxt = self._construct_multiple_contexts(example, prompt_and_fewshot)
        tokenized_example = self._tokenize_example(prompt_and_fewshot, ctxt, example)
        return tokenized_example

    def _tokenize_example(self, prompt_and_fewshot: str, context_options: List[str], example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handle special cases.

        Args:
            prompt_and_fewshot (str): the collection of the prompt and fewshot examples that belongs before the example's context
            ctx (str): the specific example's derrived context
            example (Dict): the example as a dictionary.

        Returns:
            Dict: dictionary with the tokenized data
        """
        tokenized_example = {}
        preamble = self.tokenizer(prompt_and_fewshot)
        preamble = self._fix_eos_on_preamble(preamble['input_ids'])
        encoded_contexts = [
            preamble + self.tokenizer(c, add_special_tokens=False)['input_ids'] for c in context_options
        ]
        continuation = example['continuation']
        if self.prefix_space:
            continuation = (f' {continuation}' if not continuation.startswith(' ') else continuation)
        tokenized_continuation = self.tokenizer(continuation, add_special_tokens=False)['input_ids']

        tokenized_example[self.context_key] = []
        tokenized_example['continuation_indices'] = []
        tokenized_example[self.answer_key] = []
        for context in encoded_contexts:
            trimmed_context = _trim_context(context, tokenized_continuation, self.padding_size)
            continuation_indices = _get_continuation_span(trimmed_context, tokenized_continuation)
            padded_context = _make_padded_input(trimmed_context, tokenized_continuation, self.padding_size,
                                                self.pad_tok_id, self.padding_side)
            tokenized_example[self.context_key].append(padded_context)
            tokenized_example['continuation_indices'].append(continuation_indices)
            tokenized_example[self.answer_key].append(tokenized_continuation)

        tokenized_example['gold'] = example['gold']
        return tokenized_example


class InContextLearningCodeEvalDataset(InContextLearningDataset):
    """
    A dataset that constructs batches for in-context learning code evaluation.

    The input format is expected to be a jsonl file with the following fields:
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
    - entry_points: list of entry points
    - test_inputs: list of test inputs
    - test_outputs: list of test outputs
    - languages:  list of languages
    - pass_at_k: passed value for pass_at_k
    - generation_length: derrived maximum generation length
    - generation_kwargs: Dictionary of kwargs neeeded for generation. Includes the following, which will be individually overwritten
        by keys in generaiton_kwargs if set (see https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
        for more details):
        - pad_token_id: ID for padding token, derived automatically
        - num_beams: how many beams to search for generations, set to 1
        - num_return_sequences: value passed for 'generations_per_sample', how many generations per prompt
        - do_sample: determines whether model is sampling or greedily decoding. Always set to True
        - use_cache: Whether or not to use past key values to speed up sampling. Always set to True

    Additional Args:
        generations_per_sample (int) (defaults to 1): The number of independently computed returned sequences for each element in the batch
        pass_at_k (int) (defaults to 1): k for how many chances the model gets to write passing code
    """

    def __init__(
        self,
        generations_per_sample: int,
        pass_at_k: int = 1,
        *args,
        **kwargs,
    ):
        if generations_per_sample < pass_at_k:
            raise ValueError(
                f'generations_per_sample ({generations_per_sample}) must be greater than or equal to pass_at_k ({pass_at_k}) for code evaluation.'
            )
        batch_mapping = {
            'input_ids': 'prompt',
            'prompts': 'prompt_text',
            'tests': 'test',
            'labels': 'canonical_solution',
            'entry_points': 'entry_point',
            'test_inputs': 'test_inputs',
            'test_outputs': 'test_outputs',
            'languages': 'language'
        }
        # Linting complains if this is not set in init
        self.max_prompt_length = 0
        static_keys = ['mode', 'pass_at_k', 'generation_length', 'generation_kwargs']
        list_keys = ['prompts', 'tests', 'entry_points', 'test_inputs', 'test_outputs', 'languages', 'labels']
        tensor_keys = ['input_ids', 'attention_mask']
        super().__init__(
            context_key='prompt',
            answer_key='canonical_solution',
            strip_dataset=False,
            static_keys=static_keys,
            list_keys=list_keys,
            tensor_keys=tensor_keys,
            tokenize_labels=False,
            padding_side='left',
            batch_mapping=batch_mapping,
            *args,
            **kwargs,
        )
        self.dataset = self.adjust_padding()
        self.base_batch = {
            'input_ids': [],
            'mode': 'generate',
            'labels': [],
            'prompts': [],
            'tests': [],
            'entry_points': [],
            'test_inputs': [],
            'test_outputs': [],
            'languages': [],
            'pass_at_k': pass_at_k,
            'generation_length': self.max_seq_len - self.max_prompt_length,
            'generation_kwargs': {
                'pad_token_id': self.pad_tok_id,
                'num_beams': 1,  # single beam
                'num_return_sequences': generations_per_sample,
                'do_sample': True,
                'use_cache': True
            },
        }
        self._update_generation_kwargs(kwargs.get('generation_kwargs'))

    def adjust_padding(self):
        """
        Adjusts padding to the maximum prompt length rather than max_seq_len.
        Needs to be done after the dataset has been processed because we don't know the maximum
        prompt length until after we've tokenized it.

        Returns:
            dataset: a HuggingFace Dataset with different padding lengths for example[self.context_key]
        """
        max_prompt_length = 0
        for example in self.dataset:
            unpadded_example = [token for token in example[self.context_key] if token != self.pad_tok_id]
            max_prompt_length = max(
                max_prompt_length,
                len(unpadded_example),
            )
        self.max_prompt_length = max_prompt_length

        def _trim_padding(example: Dict):
            # Remove padding tokens applied during tokenization
            unpadded_prompt = [token for token in example[self.context_key] if token != self.pad_tok_id]
            # Reapply padding only to max_prompt_length
            full_prompt = _trim_context(unpadded_prompt, [], self.max_prompt_length)
            padded_context = _make_padded_input(full_prompt, [], self.max_prompt_length, self.pad_tok_id,
                                                self.padding_side)

            example[self.context_key] = padded_context
            return example

        return self.dataset.map(_trim_padding)

    def _tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Adds extra code task details to the example dictionary.
        See InContextLearningDataset for more details
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
    generation_kwargs: Dict,
) -> DataSpec:
    """
    Factory method that builds the specific dataset for the specified icl_task_type.
    See documentation for `get_icl_task_dataloader` for arugment documentation.

    When writing a dataset for a new task, here you will need to:
        1. add the dataset to the factory and choose an appropriate string
        2. set the batch size for that task (see InContextLearningMultipleChoiceTaskDataset for why
            this might be different)
        3. set the `split_batch` funciton if necessary
    """
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
            generation_kwargs=generation_kwargs,
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
            generation_kwargs=generation_kwargs,
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
            generation_kwargs=generation_kwargs,
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
            generation_kwargs=generation_kwargs,
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
            generation_kwargs=generation_kwargs,
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
    if dataset_uri.startswith('hf://'):
        dataset_uri = dataset_uri.replace('hf://', '')
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
    prompt_string: str = '',  # e.g. 'translate english to french:'
    example_delimiter: str = '\n',  # e.g. '\n'
    continuation_delimiter: str = ' ',
    destination_path: str = '',
    question_prelimiter: str = '',  # e.g. 'Question: '
    fewshot_random_seed: int = 1234,
    pass_at_k: int = 1,
    generations_per_sample: int = 20,
    cot_delimiter: str = '',
    has_categories: bool = False,
    hf_loading_vars: Dict = None,
    hf_parsing_map: Dict = None,
    generation_kwargs: Dict = None,
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
        dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri prepended with ``hf://``.
            Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            A local dataset must consist of rows of JSON data points with task dependant fields.
            The default keys expected are "context" and "answer".
        tokenizer (transformers.PreTrainedTokenizerBase): The tokenizer used to map between strings and token ids.
        batch_size (int): Size of a batch used for eval
        max_seq_len (int): The maximum sequence length supported by the model.
        pad_tok_id (int): The special token used for padding batches.
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example. These are not identical across examples.
        prompt_string (str, default = ''): Prompt string to put once before all fewshot examples/test examples (e.g. 'Translate english to french.').
        example_delimiter (str, default = '\n'): Separator inserted before (context, answer) pairs (e.g. '\n') for fewshot sampling and prompting.
        continuation_delimiter: (str, default = ' '): Separator inserted between context and answer in each example (e.g. '\nA: ').
        destination_path: (str, default = ''): This is the local file where remote datasets will be saved.
        question_prelimiter: (str, default = ''): Text to be prepended before each context, including few shot examples (e.g. "Question: ").
        fewshot_random_seed (int, default = 1234): Random seed to use for fewshot sampling
        pass_at_k (int): k for how many chances the model gets to write passing code.
        generations_per_sample (int): how many outputs to generate per prompt. Passed in generation_kwargs under "num_return_sequences" and overwritten by generation_kwargs dict.
        cot_delimiter (str): Delimiter to place between chain of thoughts and continuations.
        has_categories: (bool): If ``True``, we will search the dataset file for a category key, and partition the dataset into a separate dataloader for each category occurring in the data.
        hf_loading_vars (Dict, default = None): A dictionary containing keyword arguments to be passed into `load_dataset` if dataset is being pulled from HF.
        hf_parsing_map (Dict, default = None): A dictionary containing a mapping from HF columns to ICL dataset keys. The dictionary should be formatted {icl_key:[hf_key1, hf_key1]}.
            Column contents will be concatenated with ' ' seperating them. If not included, will load the columns already present in the HF dataset.
        generation_kwargs (Dict, default = None): A dictionary containing keyword arguments to be passed along to the model's generate function. Overwrites any previously specified generation
                                                  keyword args in this fucntion (see https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
                                                  for more details)

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
                generation_kwargs=generation_kwargs,
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
            generation_kwargs=generation_kwargs,
        )
