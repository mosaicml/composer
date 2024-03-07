# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for datasets."""

import logging
import textwrap
from typing import Callable, List, Optional

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

__all__ = [
    'add_vision_dataset_transform',
    'MultiTokenEOSCriteria',
]

log = logging.getLogger(__name__)


def add_vision_dataset_transform(dataset: VisionDataset, transform: Callable, is_tensor_transform: bool = False):
    """Add a transform to a dataset's collection of transforms.
    Args:
        dataset (VisionDataset): A torchvision dataset.
        transform (Callable): Function to be added to the dataset's collection of
            transforms.
        is_tensor_transform (bool): Whether ``transform`` acts on data of the type
            :class:`~torch.Tensor`. default: ``False``.
            * If ``True``, and :class:`~torchvision.transforms.ToTensor` is present in the transforms of the
              ``dataset``, then ``transform`` will be inserted after the
              :class:`~torchvision.transforms.ToTensor` transform.
            * If ``False`` and :class:`~torchvision.transforms.ToTensor` is present, the ``transform`` will be
              inserted before :class:`~torchvision.transforms.ToTensor`.
            * If :class:`~torchvision.transforms.ToTensor` is not present, the transform will be appended to
              the end of collection of transforms.
    Returns:
        None: The ``dataset`` is modified in-place.
    """
    transform_added_logstring = textwrap.dedent(
        f"""Transform {transform} added to dataset.
        Dataset now has the following transforms: {dataset.transform}""",
    )

    if dataset.transform is None:
        dataset.transform = transform
        log.warning(transform_added_logstring)
    elif isinstance(dataset.transform, transforms.Compose):
        insertion_index = len(dataset.transform.transforms)
        for i, t in enumerate(dataset.transform.transforms):
            if isinstance(t, transforms.ToTensor):
                insertion_index = i
                break
        if is_tensor_transform:
            insertion_index += 1
        dataset.transform.transforms.insert(insertion_index, transform)
        log.warning(transform_added_logstring)
    else:  # transform is some other basic transform, join using Compose
        if isinstance(dataset.transform, transforms.ToTensor) and not is_tensor_transform:
            dataset.transform = transforms.Compose([transform, dataset.transform])
            log.warning(transform_added_logstring)
        else:
            dataset.transform = transforms.Compose([dataset.transform, transform])
            log.warning(transform_added_logstring)


try:
    import transformers

    class MultiTokenEOSCriteria(transformers.StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence.
        Slightly modified from: https://github.com/EleutherAI/lm-evaluation-harness/blob/78545d42f2ca95c6fe0ed220d456eeb94f4485e9/lm_eval/utils.py#L614-L649
        """

        def __init__(
            self,
            stop_sequence: str,
            tokenizer: transformers.PreTrainedTokenizerBase,
            batch_size: int,
        ) -> None:
            self.done_tracker = [False] * batch_size
            self.stop_sequence = stop_sequence
            self.stop_sequence_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)

            # sentence piece tokenizers add a superflous underline token before string-initial \n
            # that throws off our calculation of the stop sequence length
            # so we remove any token ids that produce empty strings
            self.stop_sequence_ids = [id for id in self.stop_sequence_ids if tokenizer.decode(id) != '']

            # we look back for 1 more token than it takes to encode our stop sequence
            # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
            # and we don't want to mistakenly not stop a generation because our
            # (string) stop sequence was output in a different tokenization

            self.stop_sequence_id_len = len(self.stop_sequence_ids) + 1
            self.tokenizer = tokenizer

        def __call__(self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None, **kwargs) -> bool:
            # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
            lookback_ids_batch = input_ids[:, :][:, -self.stop_sequence_id_len:]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if i >= len(lookback_tokens_batch):
                    # The last batch of a dataset may be smaller than `batch_size`
                    # Automatically set those indices in the done_tracker to True
                    # since those indices don't show up in the current batch
                    self.done_tracker[i] = True
                    break
                elif not done:
                    self.done_tracker[i] = self.stop_sequence in lookback_tokens_batch[i]
            return False not in self.done_tracker

    def stop_sequences_criteria(
        tokenizer: transformers.PreTrainedTokenizerBase,
        stop_sequences: List[str],
        batch_size: int,
    ) -> transformers.StoppingCriteriaList:
        return transformers.StoppingCriteriaList([
            *[MultiTokenEOSCriteria(sequence, tokenizer, batch_size) for sequence in stop_sequences],
        ])

except ImportError as e:
    stop_sequences_criteria = None  # pyright: ignore [reportGeneralTypeIssues]
    MultiTokenEOSCriteria = None  # pyright: ignore [reportGeneralTypeIssues]
