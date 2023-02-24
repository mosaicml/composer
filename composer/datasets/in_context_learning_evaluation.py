# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# This code is based on the implementation in https://github.com/EleutherAI/lm-evaluation-harness/blob/8c048e266a22a1c85ccbdb0c209ac712e4f39989/lm_eval/base.py#L221-L330

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Union

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from composer.core import DataSpec
from composer.utils import MissingConditionalImportError, dist, get_file

if TYPE_CHECKING:
    import transformers

__all__ = ['InContextLearningLMTaskDataset', 'InContextLearningMultipleChoiceTaskDataset', 'get_icl_task_dataloader']


def _make_padded_input(context_enc, continuation_enc, max_seq_len, pad_tok_id):
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
    inp = torch.cat(
        [
            inp,  # [seq]
            torch.LongTensor((max_seq_len - inp_len) * [pad_tok_id]),
        ],
        dim=0,
    )

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
    ):
        try:
            from datasets import load_dataset  # pyright: ignore [reportGeneralTypeIssues]
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='datasets',
                                                conda_channel='conda-forge') from e
        with dist.local_rank_zero_download_and_wait(destination_path):
            get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        self.samples = list(
            dataset.map(lambda examples: {
                'continuation': examples['continuation'],
                'context': examples['context'],
            }))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.encoded_dataset = self.prep_examples(num_fewshot, prompt_string, example_delimiter, continuation_delimiter)

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
            encoded_example = {}

            preamble = prompt_string

            if num_fewshot > 0:
                fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx)
                for fewshot_idx in fewshot_idxs:
                    ctxt, cont = self.samples[fewshot_idx]['context'], self.samples[fewshot_idx]['continuation']
                    if len(preamble) > 0:
                        ctxt = f'{example_delimiter}{ctxt}'
                    preamble += f'{ctxt}{continuation_delimiter}{cont}'

            ctxt, cont = self.samples[sample_idx]['context'], self.samples[sample_idx]['continuation']
            if len(preamble) > 0:
                ctxt = f'{example_delimiter}{ctxt}'

            cont = f'{continuation_delimiter}{cont}'

            encoded_example['preamble'] = self.tokenizer(
                preamble
            )  # if the preamble is empty then these will be 0-length lists, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
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
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        destination_path (str): Temporary path to store downloaded datasets
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
    ):
        try:
            from datasets import load_dataset  # pyright: ignore [reportGeneralTypeIssues]
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='datasets',
                                                conda_channel='conda-forge') from e

        with dist.local_rank_zero_download_and_wait(destination_path):
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

            preamble = prompt_string
            if num_fewshot > 0:
                fewshot_idxs = _get_fewshot_sample_idxs(len(self.samples), num_fewshot, sample_idx)
                for fewshot_idx in fewshot_idxs:
                    query, choices, gold_idx = self.samples[fewshot_idx]['query'], self.samples[fewshot_idx][
                        'choices'], self.samples[fewshot_idx]['gold']
                    if len(preamble) > 0:
                        query = f'{example_delimiter}{query}'
                    preamble += f'{query}{continuation_delimiter}{choices[gold_idx]}'

            encoded_example = {}
            query, choices, gold_idx = self.samples[sample_idx]['query'], self.samples[sample_idx][
                'choices'], self.samples[sample_idx]['gold'],
            if len(preamble) > 0:
                query = f'{example_delimiter}{query}'
            choices = [f'{continuation_delimiter}{choice}' for choice in choices]
            encoded_example['preamble'] = self.tokenizer(
                preamble
            )  # if the preamble is empty then these will be 0-length lists, unless the tokenizer adds special tokens to empty strings (e.g. OPT tokenizer)
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
        return batch['input_ids'].shape[0]

    def split_batch(self, batch: Any, microbatch_size: int):
        raise Exception(f"""We haven't implemented batch splitting for multiple choice tasks""")


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
) -> DataSpec:
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
                                                             destination_path=destination_path)
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
                                                 destination_path=destination_path)
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
