# Copyright 2021 MosaicML. All Rights Reserved.

import textwrap
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core.types import Algorithm, Batch, Event, Logger, State, Tensor
from composer.models.transformer_shared import ComposerTransformer
from composer.utils import ensure_tuple


def apply_seq_length_warmup(batch: Dict[str, Tensor], curr_seq_len: int, truncate: bool) -> Batch:
    """Progressively increases the sequence length during training.

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
class SeqLengthWarmupHparams(AlgorithmHparams):

    duration: float = hp.optional("Fraction of total training time to apply sequential length warmup learning.",
                                  default=0.3)
    min_seq_length: int = hp.optional("Starting sequence length.", default=8)
    max_seq_length: int = hp.optional("End sequence length", default=1024)
    step_size: int = hp.optional("Sequence length step size", default=8)
    truncate: bool = hp.optional("Truncate tensors or reshape extra tokens to new examples.", default=True)

    def initialize_object(self) -> "SeqLengthWarmup":
        return SeqLengthWarmup(**asdict(self))


class SeqLengthWarmup(Algorithm):
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
        we follow PyTorch notes and pre-allocate the memory with a blank
        forward and backward pass.

    Args:
        duration (float): fraction of total training for sequential length learning.
        min_seq_length (int): Minimum sequence length to start the warmup.
        max_seq_length (int): Maximum sequence length to stop the warmup.
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
        self.duration = duration
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.step_size = step_size
        self.truncate = truncate

        if self.duration < 0 or self.duration > 1:
            raise ValueError(f'Duration must be getween 0 and 1, got: {self.duration}')

        if self.max_seq_length < self.min_seq_length:
            raise ValueError(f'max_seq_length={self.max_seq_length} must be '
                             f'greater than min_seq_length={self.min_seq_length}')
        self._activated = False
        self._original_model = None

    def match(self, event: Event, state: State) -> bool:
        return event in (Event.INIT, Event.AFTER_DATALOADER)

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Applies on ``Event.AFTER_DATALOADER`` to apply the sequence length warmup to the input batch.

        .. note::

            On the first call of :meth:`apply`, a dummy training pass on the
            full sequence length is used to preallocate the PyTorch cache.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
            logger (:class:`Logger`): A logger to use for logging algorithm-specific metrics.
        Returns:
            int or None: exit code that is stored in :class:`Trace` and made accessible for debugging.
        """
        if event == Event.INIT:
            if self._original_model is None:
                if not isinstance(state.model, ComposerTransformer):
                    raise RuntimeError(
                        textwrap.dedent(f"""\
                        {type(self).__name__} requires state.modelto be of type {ComposerTransformer.__name__},
                        not of type {type(state.model)}"""))
                self._original_model = state.model
            return

        # in order to avoid OOMs, we do a forward and a backward pass on a dummy input.
        if not self._activated:
            # ensure that input_ids is a valid model input. since we don't need the
            # results, we don't use all inputs.
            assert self._original_model is not None, "original model should be set on Event.INIT"
            model_inputs = self._original_model.get_model_inputs()
            if 'input_ids' not in model_inputs:
                raise RuntimeError("'input_ids' must be in model inputs")
            if 'labels' not in model_inputs:
                raise RuntimeError("'labels' must be in model inputs")

            # create fake inputs
            vocab_size = len(self._original_model.tokenizer)

            # simplifying assumption: Composer doesn't support model-parallelism,
            # so the first parameter's device is likely the same device for
            # all of the parameters
            device = next(state.model.parameters()).device

            per_gpu_macrobatch = state.train_dataloader.batch_size
            if per_gpu_macrobatch is None:
                raise RuntimeError("seq_length_warmup requires constant batch sizing")
            assert per_gpu_macrobatch % state.grad_accum == 0, "grad accum should evenly divide the batch"
            per_gpu_batch = per_gpu_macrobatch // state.grad_accum

            input_ids = torch.randint(low=0,
                                      high=vocab_size - 1,
                                      size=(per_gpu_batch, self.max_seq_length),
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
            with state.precision_context:
                outputs = state.model.forward(model_inputs)
                loss = self._original_model.loss(outputs, model_inputs)

            # since use_grad_scaling is in the Trainer, and we
            # don't care about the loss values, skip scaling
            for loss_item in ensure_tuple(loss):
                loss_item.backward()

            for optimizer in state.optimizers:
                optimizer.zero_grad()

            self._activated = True

        num_optimization_steps = state.steps_per_epoch * state.max_epochs
        num_warmup_steps = int(num_optimization_steps * self.duration)

        # assume the full sequence length is the unaltered sequence length
        num_update_steps = (self.max_seq_length - self.min_seq_length) // self.step_size
        update_every_n_steps = num_warmup_steps // num_update_steps

        curr_seq_len = self.step_size * (state.step // update_every_n_steps)
        curr_seq_len = max(curr_seq_len, self.min_seq_length)
        curr_seq_len = min(curr_seq_len, self.max_seq_length)

        state.batch = apply_seq_length_warmup(state.batch_dict, curr_seq_len, self.truncate)

        batch_size = state.batch_dict['input_ids'].shape[0]
        logger.metric_batch({
            'seq_length_warmup/curr_seq_len': curr_seq_len,
            'seq_length_warmup/curr_bs': batch_size,
        })
