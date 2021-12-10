# Copyright 2021 MosaicML. All Rights Reserved.

import math
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core.types import Algorithm, Batch, Event, Logger, State, Tensor
from composer.models.transformer_shared import MosaicTransformer
from composer.utils import ddp, ensure_tuple


def apply_curriculum(batch: Dict[str, Tensor], curr_seq_len: int, truncate: bool) -> Batch:
    """
    Progressively increases the sequence length during training.

    Changes the sequence length of all tensors in the provided dictionary
    to ``curr_seq_len``, by either truncating the tensors (``truncate=True``)
    or reshaping the tensors to create new examples from the extra tokens
    (``truncate=False``).

    The schedule for ``curr_seq_len`` over training time should be managed
    out of this function.

    Args:
        batch: The input batch to the model, must be a dictionary.
        curr_seq_length (int): The desired sequence length to apply.
        truncate (bool): Truncate sequences early, or reshape tensors
                         to create new examples out of the extra tokens.

    Returns:
        batch: a Mapping of input tensors to the model,
               where all tensors have curr_seq_len in the second dimension.
    """

    assert isinstance(batch, Mapping)

    if truncate:
        for k in batch.keys():
            batch[k] = batch[k][:, :curr_seq_len]
    else:
        # ensure new tensor shape is divisible by curr_seq_len
        input_ids = batch['input_ids'].view(-1)
        tensor_len = (input_ids.shape[0] // curr_seq_len) * curr_seq_len

        input_ids = input_ids[:tensor_len]
        input_ids = input_ids.view(-1, curr_seq_len)
        batch['input_ids'] = input_ids

        for k, v in batch.items():
            if k == "input_ids":
                continue
            v = v.view(-1)
            v = v[:tensor_len]
            batch[k] = v.view(-1, curr_seq_len)

    return batch


@dataclass
class CurriculumLearningHparams(AlgorithmHparams):

    duration: float = hp.optional("Fraction of total training time to apply curriculum learning.", default=0.3)
    min_seq_length: int = hp.optional("Starting sequence length.", default=8)
    max_seq_length: int = hp.optional("End sequence length", default=1024)
    step_size: int = hp.optional("Sequence length step size", default=8)
    truncate: bool = hp.optional("Truncate tensors or reshape extra tokens to new examples.", default=True)

    def validate(self):
        if self.duration < 0 or self.duration > 1:
            raise ValueError(f'Duration must be getween 0 and 1, got: {self.duration}')

        if self.max_seq_length < self.min_seq_length:
            raise ValueError(f'max_seq_length={self.max_seq_length} must be '
                             f'greater than min_seq_length={self.min_seq_length}')

    def initialize_object(self) -> "CurriculumLearning":
        return CurriculumLearning(**asdict(self))


class CurriculumLearning(Algorithm):
    """Progressively increases the sequence length during training.

    Changes the sequence length of all tensors in the input batch. The
    sequence length increases from ``min_seq_length`` to ``max_seq_length``
    in steps of ``step_size`` during the first ``duration`` fraction of
    training.

    The sequence length is then kept at ``max_seq_length``
    for the rest of training.

    Tensors are either truncated (``truncate=True``) or reshaped to
    create new examples from the extra tokens (``truncate=False``).

    .. note::

        ``step_size`` should be a multiple of eight for GPUs

    .. note::

        Variable input lengths can create CUDA OOM errors. To avoid this,
        we follow pytorch notes and pre-allocate the memory with a blank
        forward and backward pass.

    Args:
        duration (float): fraction of total training for curriculum learning.
        min_seq_length (int): Minimum sequence length to start the curriculum.
        max_seq_length (int): Maximum sequence length to stop the curriculum.
        step_size (int): Step size of sequence length.

        truncate (bool): Truncate tensors or reshape extra tokens to new examples
    """

    def __init__(
        self,
        duration: float = 0.3,
        min_seq_length: int = 8,
        max_seq_length: int = 1024,
        step_size: int = 8,
        truncate: bool = True,
    ):
        self.hparams = CurriculumLearningHparams(duration=duration,
                                                 min_seq_length=min_seq_length,
                                                 max_seq_length=max_seq_length,
                                                 step_size=step_size,
                                                 truncate=truncate)
        self.hparams.validate()

    def match(self, event: Event, state: State) -> bool:
        return event in (Event.TRAINING_START, Event.AFTER_DATALOADER)

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:

        # in order to avoid OOMs, we do a forward and a backward pass on a dummy input.
        if event == Event.TRAINING_START:
            # ensure that input_ids is a valid model input. since we don't need the
            # results, we don't use all inputs.

            original_model = state.model.module
            assert isinstance(original_model, MosaicTransformer)
            model_inputs = original_model.get_model_inputs()  # type: ignore
            assert 'input_ids' in model_inputs
            assert 'labels' in model_inputs

            # create fake inputs
            vocab_size = len(original_model.tokenizer)  # type: ignore

            # simplifying assumption: Composer doesn't support model-parallelism,
            # so the first parameter's device is likely the same device for
            # all of the parameters
            device = next(state.model.parameters()).device

            assert (state.train_batch_size % ddp.get_world_size()) == 0
            per_gpu_batch = math.ceil(state.train_batch_size / (ddp.get_world_size() * state.grad_accum))
            input_ids = torch.randint(low=0,
                                      high=vocab_size - 1,
                                      size=(per_gpu_batch, self.hparams.max_seq_length),
                                      device=device).long()
            labels = input_ids.clone()
            attn_mask = torch.ones_like(labels)
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attn_mask,
            }

            # start by running a forward and backward pass
            # of the maximum sequence length to allocate cache.
            with state.precision_context(state.precision):
                outputs = state.model.forward(model_inputs)
                loss = original_model.loss(outputs, model_inputs)

            # since use_grad_scaling is in the Trainer, and we
            # don't care about the loss values, skip scaling
            for loss_item in ensure_tuple(loss):
                loss_item.backward()

            # zero out gradients and proceed to normal training
            assert state.optimizers is not None, \
                "optimizers are set before TRAINING_START"

            for optimizer in ensure_tuple(state.optimizers):
                optimizer.zero_grad()
        else:
            num_optimization_steps = state.steps_per_epoch * state.max_epochs
            num_curriculum_steps = int(num_optimization_steps * self.hparams.duration)

            # assume the full sequence length is the unaltered sequence length
            num_update_steps = (self.hparams.max_seq_length - self.hparams.min_seq_length) // self.hparams.step_size
            update_every_n_steps = num_curriculum_steps // num_update_steps

            curr_seq_len = self.hparams.step_size * (state.step // update_every_n_steps)
            curr_seq_len = max(curr_seq_len, self.hparams.min_seq_length)
            curr_seq_len = min(curr_seq_len, self.hparams.max_seq_length)

            state.batch = apply_curriculum(state.batch_dict, curr_seq_len, self.hparams.truncate)

            batch_size = state.batch_dict['input_ids'].shape[0]
            logger.metric_batch({
                'curriculum_learning/curr_seq_len': curr_seq_len,
                'curriculum_learning/curr_bs': batch_size,
            })
