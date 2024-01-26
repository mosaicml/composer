# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import torch
from torch.utils.data import Dataset

from composer.core.data_spec import _default_split_batch, _split_list
from composer.utils import MissingConditionalImportError, dist, get_file

if TYPE_CHECKING:
    import transformers
    from datasets import Dataset as HFDataset  # pyright: ignore[reportGeneralTypeIssues]


__all__ = [
    'InContextLearningDataset',
]


def strip_data(example: Dict) -> Dict:
    """
    Remove white space from the begging and end of string values in a dictionary

    Args:
        example: Dictionary to be stripped

    Returns:
        dict: The same dictionary with .strip() applied to any value in the dict that is a string
    """
    return {k: v.strip() if isinstance(v, str) else v for k, v in example.items()}


def _tokenizer_needs_prefix_space(tokenizer: transformers.PreTrainedTokenizerBase) -> bool:
    """
    Test for whether a prefix space is needed before the continuation.
    Sentencepiece tokenization should not have a prefix space, but gpt2 style BPE should.

    Args:
        tokenizer: Tokenizer to test

    Returns:
        bool: Whether or not the tokenizer needs a prefix space
    """
    test_tokens = tokenizer(' a', add_special_tokens=False)['input_ids']
    assert isinstance(test_tokens, list)
    return len(test_tokens) == 1


def _trim_context(context_enc: List, continuation_enc: List, max_seq_len: int) -> List:
    """
    Trims a list of tokens down to `max_seq_len` if the length of the list plus the continuation
    is more than `max_seq_len`. It will always trim tokens from the left, i.e. tokens at the beginning
    of the context will be removed.

    Args:
        context_enc (list): List of tokens in the context
        continuation_enc (lsit): List of tokens in the continuation
        max_seq_len (int): Maximum length the model can ingest

    Returns:
        list: The encoded context trimmed from the left
    """
    if len(continuation_enc) + len(context_enc) > max_seq_len:
        context_max_subseq_len = max_seq_len - len(continuation_enc)

        if context_max_subseq_len < 0:
            # can't support continuations which are longer than the max seq len
            raise Exception(f'Dataset included continuation longer than the max seq len')

        # clip from the end
        context_enc = context_enc[-(context_max_subseq_len):]
    return context_enc


def _get_continuation_span(context_enc: List, continuation_enc: List) -> torch.Tensor:
    """
    Gets the list of indices of the continuation tokens for language modeling or generation tasks.

    Args:
        context_enc (list): List of context tokens
        continuation_enc (list): List of continuation tokens

    Returns:
        torch.tensor: A tensor containing indices corresponding to continuation tokens
    """
    return torch.tensor(range(len(context_enc), len(context_enc) + len(continuation_enc)))


def _make_padded_input(context_enc: List,
                       continuation_enc: List,
                       max_seq_len: int,
                       pad_tok_id: int,
                       padding_side: str = 'right') -> torch.Tensor:
    """
    Takes an encoded context and continuation and clips the beginning of the context if they're too long.
    Adds the padding token to the specified side.

    Args:
        context_enc (List): The encoded input to the model
        continuation_enc (List): The encoded desired output for the example
        max_seq_list (int): Maximum length sequences can be
        pad_tok_id (int): The token id we pad with
        padding_side (str): Which side to pad the context on. Can be 'right' or 'left

    Returns:
        input (torch.tensor): The padded and encoded context
        continuation_span (torch.tensor): The _inclusive_ range of indices corresponding to the continuation
    """

    inp = torch.tensor(
        (context_enc + continuation_enc),
        dtype=torch.long,
    )
    (inp_len,) = inp.shape

    # Sometimes tokenizers that have neither a pad_tok_id or eos_tok_id will pass None in as the padding
    # token and cause errors
    if not isinstance(pad_tok_id, int):
        raise ValueError(f'`pad_tok_id` must be an integer. Found {type(pad_tok_id)} instead')
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
        batch (dict): A dictionary of batched inputs
        tokenize_labels (bool): Whether or not the labels are tokenized (and need to be stacked)

    Returns:
        dict: The batch with torch tensors in the corresponding keys instead of lists of lists
    """
    batch['input_ids'] = torch.stack(list(map(torch.tensor, batch['input_ids'])))
    if tokenize_labels:
        batch['labels'] = torch.stack(list(map(torch.tensor, batch['labels'])))
        batch['continuation_indices'] = list(map(torch.tensor, batch['continuation_indices']))
    return batch


def _get_fewshot_sample_idxs(dataset_size: int, num_fewshot: int, example_idx: int, rng: random.Random) -> Set[int]:
    """
    Samples indices without replacement. If num_fewshot exceeds the number of unique examples in the dataset,
    then we will have fewer than num_fewshot examples in context.
    Args:
        dataset_size (int): Length of the dataset
        num_fewshot (int): Number of examples to prepend
        example_idx (int): Current example's index (excluded from fewshot choices)
        rng (random.Random): RNG for repeatable sample selection

    Returns:
        list: Indices of the examples chosen for fewshot selection
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
    'example' refers to a loaded dictionary, generally containing a context, an answer, and any other information needed to run the task.
    'answer' refers to the desired output of the model.

    When creating a new ICL Dataset, it is likely that you will need to reimplement the following methods:

    - construct_context(): Takes a single example dictionary and formulates the context as a string for that eval question.
    - get_answer_from_example(): Takes a single example dictionary and formulates the correct, ground truth answer as a string.
    - tokenize_example(): Tokenizes the example and adds any extra content from the original dictionary that needs to be passed downstream.
    - read_dataset(): Loads the dataset and does basic parsing. If additional parsing must be done, this is a good place to do so (See InContextLearningQATaskDataset.read_dataset())

    Additionally, base_batch and batch_mapping must be defined.

    - base_batch (Dict): The base dictionary that the dataset will use to construct a batch. This should contain static values, like generation_kwargs or mode,
      and empty lists for values that will need to be accumulated from each example.
      NOTE: Sometimes you will need to set base_batch directly after the init call, e.g. in order to use class variables
      like self.pad_tok_id or self.max_answer_length. If you manually set generation_kwargs this way, you'll need to call self.update_generation_kwargs()
      after setting self.base_batch.
    - batch_mapping (Dict): A mapping with keys that are keys in the batch and values that are columns in the loaded dataset.
      collate_fn will use this mapping to create batches from self.dataset.

    Args:
        dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri prepended with ``hf://``.
            Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            A local dataset must consist of rows of JSON data points with task dependent fields.
            The default keys expected are "context" and "answer".
        tokenizer (transformers.PreTrainedTokenizerBase): The tokenizer used to map between strings and token ids.
        max_seq_len (int): The maximum sequence length supported by the model.
        pad_tok_id (int): The special token used for padding batches.
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example. These are not identical across examples.
        fewshot_random_seed (int): Random seed to use for fewshot sampling.
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples (e.g. 'Translate english to french.').
        example_delimiter (str): Separator inserted before (context, answer) pairs (e.g. '\\n') for fewshot sampling and prompting.
        continuation_delimiter: (str): Separator inserted between context and answer in each example (e.g. '\\nA: ').
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
        prelimiter: str = '',
        context_key: str = 'context',
        answer_key: str = 'answer',
        strip_dataset: bool = True,
        padding_side: str = 'right',
        tokenize_labels: bool = True,
        static_keys: Optional[List] = None,
        list_keys: Optional[List] = None,
        tensor_keys: Optional[List] = None,
        padding_size: Optional[int] = None,
        base_batch: Optional[Dict] = None,
        batch_mapping: Optional[Dict] = None,
        hf_loading_vars: Optional[Dict] = None,
        hf_parsing_map: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None,
    ):
        try:
            import datasets
            del datasets
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='nlp',
                conda_package='datasets',
                conda_channel='conda-forge',
            ) from e

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
        self.update_generation_kwargs(generation_kwargs or {})

        self.static_keys = static_keys
        self.list_keys = list_keys
        self.tensor_keys = tensor_keys

        hf_loading_vars = hf_loading_vars or {}
        self.dataset: HFDataset = self.read_dataset(dataset_uri, destination_path, hf_loading_vars, hf_parsing_map)
        self.strip_data = strip_dataset
        if self.strip_data:
            self.dataset = self.dataset.map(strip_data)

        fewshot_rng = random.Random(fewshot_random_seed)
        self.dataset: HFDataset = self.dataset.map(
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

    def update_generation_kwargs(self, generation_kwargs: Dict) -> None:
        """
        Updates self.base_batch with the passed in generation_kwargs.
        This must be run after self.base_batch is set (for example, if self.base_batch is set after __init__() is run,
        likely because base_batch needs a class variable like self.pad_tok_id or self.max_answer_length).

        Args:
            dict: Keyword arguments that be written into base_batch['generation_kwargs']
        """
        if 'generation_kwargs' not in self.base_batch:
            self.base_batch['generation_kwargs'] = {}
        if generation_kwargs:
            self.base_batch['generation_kwargs'].update(generation_kwargs)

    def read_dataset(self,
                     dataset_uri: str,
                     destination_path: str,
                     hf_loading_vars: Optional[Dict[str, Any]] = None,
                     hf_parsing_map: Optional[Dict[str, Any]] = None) -> 'HFDataset':
        """
        Reads a dataset and handles parsing it from HuggingFace.

        Args:
            dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri.
                Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            destination_path (str): A local path where the data will be stored
            hf_loading_vars (Dict): If parsing from HuggingFace, keyword args that will be passed into load_dataset
            hf_parsing_map (Dict): Dictionary in the form of {icl_key: [hf_col1, hf_col2]} that will map one or more hf columns, in order, to ICL dataset columns

        Returns:
            dataset: A loaded HF dataset
        """
        from datasets import Dataset as HFDataset  # pyright: ignore[reportGeneralTypeIssues]
        from datasets import load_dataset  # pyright: ignore[reportGeneralTypeIssues]
        if 'hf://' in dataset_uri:
            dataset_uri = dataset_uri.replace('hf://', '')
            if hf_loading_vars is None:
                hf_loading_vars = {}
            dataset = load_dataset(dataset_uri, **hf_loading_vars)
            if hf_parsing_map:
                dataset_parsing_func = lambda example: {
                    k: ' '.join([str(example[col]) for col in v])
                    for k, v in hf_parsing_map.items()  # pyright: ignore[reportOptionalMemberAccess]
                }
                assert isinstance(dataset, HFDataset)
                dataset = dataset.map(dataset_parsing_func, remove_columns=dataset.column_names)
        else:
            with dist.local_rank_zero_download_and_wait(destination_path):
                if dist.get_local_rank() == 0:
                    get_file(dataset_uri, destination_path, overwrite=True)
            dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        assert isinstance(dataset, HFDataset)
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
            num_fewshot (int): Number of examples to prepend
            example_idx (int): Current example idx
            preamble (str): Text to occur at the beginning of the task. Generally instructions or a prompt.
            fewshot_rng (random.Random): Seeded sampler to chose samples with

        Returns:
            str: The original preamble with num_fewshot examples appended
        """
        few_shot_text = preamble

        if num_fewshot > 0:
            fewshot_idxs = _get_fewshot_sample_idxs(
                len(self.dataset),
                num_fewshot,
                example_idx,
                fewshot_rng,
            )
            for fewshot_idx in fewshot_idxs:
                ctxt = self.construct_context(
                    self.dataset[fewshot_idx],
                    few_shot_text,
                    add_answer=True,
                )
                few_shot_text += ctxt

        return few_shot_text

    def construct_context(self, example: Dict, preceding_text: str = '', add_answer: bool = False) -> str:
        """
        Takes an example and constructs a context, i.e. the input the model reads for this example.
        Optionally adds the correct answer (for fewshot examples) and handles example delimiters

        Args:
            example (Dict): The example from which to construct the context
            preceding_text (str): Any preceding text, used as a check for prepending self.example_delimiter
            add_answer (bool): Bool for whether or not to add the answer on the end of the context (e.g. for fewshot examples)

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
            ctxt = f'{ctxt}{self.get_answer_from_example(example, in_context=add_answer)}'
        return ctxt

    def get_answer_from_example(self, example: Dict[str, Any], in_context: bool = False) -> str:
        """
        Returns the answer from the example.

        Args:
            example (Dict): The example from which to retrieve the answer

        Returns:
            str: The answer in the example
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
            input_ids (List): The tokenized input

        Returns:
            input_ids: The tokenized input conditionally edited
        """
        if (self.tokenizer.eos_token_id is not None and len(input_ids) > 1 and
                input_ids[-1] == self.tokenizer.eos_token_id):
            input_ids = input_ids[:-1]
        return input_ids

    def tokenize_example(self, prompt_and_fewshot: str, ctxt: str, example: Dict) -> Dict[str, Any]:
        """
        Runs text through the tokenizer and handle special cases.

        Args:
            prompt_and_fewshot (str): The collection of the prompt and fewshot examples that belongs before the example's context
            ctxt (str): The specific example's derrived context
            example (Dict): The example as a dictionary. Used for additional processing in inherited classes.

        Returns:
            Dict: Dictionary with the tokenized data
        """
        tokenized_example = {}
        # Always add special tokens to preamble
        preamble = self.tokenizer(prompt_and_fewshot)['input_ids']
        assert isinstance(preamble, list)
        preamble = self._fix_eos_on_preamble(preamble)
        if self.strip_data:
            # rstrip context because a prompt ending in a space results in degenerate output
            ctxt = ctxt.rstrip()
        # Never add special tokens to context
        tokenized_context = self.tokenizer(ctxt, add_special_tokens=False)['input_ids']
        assert isinstance(preamble, list)
        assert isinstance(tokenized_context, list)

        tokenized_context = preamble + tokenized_context

        if self.tokenize_labels:
            # Never add special tokens to answer
            tokenized_answer = self.tokenizer(self.get_answer_from_example(example),
                                              add_special_tokens=False)['input_ids']
            assert isinstance(tokenized_answer, list)
            trimmed_context = _trim_context(tokenized_context, tokenized_answer, self.padding_size)
            assert isinstance(trimmed_context, list)
            continuation_indices = _get_continuation_span(trimmed_context, tokenized_answer)
            padded_context = _make_padded_input(trimmed_context, tokenized_answer, self.padding_size, self.pad_tok_id,
                                                self.padding_side)

            tokenized_example[self.context_key] = padded_context
            tokenized_example[self.answer_key] = tokenized_answer
            tokenized_example['continuation_indices'] = continuation_indices
        else:
            assert isinstance(tokenized_context, list)
            trimmed_context = _trim_context(
                tokenized_context,
                [],
                self.padding_size,
            )
            assert isinstance(trimmed_context, list)
            padded_context = _make_padded_input(trimmed_context, [], self.padding_size, self.pad_tok_id,
                                                self.padding_side)

            tokenized_example[self.context_key] = padded_context
            tokenized_example[self.answer_key] = self.get_answer_from_example(example)

        return tokenized_example

    def _prep_example(
        self,
        example: Dict,
        example_idx: int,
        num_fewshot: int,
        prompt_string: str,
        fewshot_rng: random.Random,
    ) -> Dict[str, Any]:
        """
        Prepares a single example from a HF Dataset into tokenized format with prompt and fewshot examples.

        Each task consists of a context and a continuation as well as an optional prompt and optional list of
        example context/continuation pairs which precede the test context/continuation pair.

        Args:
            example (Dict): A Dictionary from the hf dataset
            example_idx (int): The index of example
            num_fewshot (int): Number of examples context/continuation pairs to prepend to the test pair
            prompt_string (str): The prompt to prepend to all inputs
            fewshot_rng (random.Random): Random number generator to use for fewshot sampling

        Returns:
            Dict: Contains a dictionary with the tokenized data
        """
        prompt_and_fewshot = self._generate_few_shot_prompt(num_fewshot, example_idx, prompt_string, fewshot_rng)
        ctxt = self.construct_context(example, prompt_and_fewshot, add_answer=False)
        tokenized_example = self.tokenize_example(prompt_and_fewshot, ctxt, example)
        return tokenized_example

    def collate_fn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        The function that the dataloader uses to accumulate data into batches.

        Args:
            data (List): List of tokenized datapoints (dicts returned by self._tokenize_example)

        Returns:
            Dict: Dictionary for a single batch
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
            batch (Dict): Batch of data
            microbatch_size (int): Size of microbatches

        Returns:
            List: List of chunked batches
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
