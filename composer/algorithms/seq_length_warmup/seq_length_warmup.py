# Copyright 2021 MosaicML. All Rights Reserved.

"""Core code for sequence length warmup."""

import textwrap
from math import ceil
from typing import Dict, Mapping, Optional

import torch

from composer.core.time import TimeUnit
from composer.core.types import Algorithm, Batch, Event, Logger, State, Tensor
from composer.models.transformer_shared import ComposerTransformer
from composer.utils import ensure_tuple

__all__ = ["SeqLengthWarmup", "set_batch_sequence_length"]


def set_batch_sequence_length(batch: Dict[str, Tensor], curr_seq_len: int, truncate: bool = True) -> Batch:
    """Set the sequence length of a batch.

    Changes the sequence length of all tensors in the provided dictionary
    to ``curr_seq_len``, by either truncating the tensors (``truncate=True``)
    or reshaping the tensors to create new examples from the extra tokens
    (``truncate=False``).

    .. note::

        The schedule for ``curr_seq_len`` over training time should be managed
        outside of this function.

    .. note::

        Variable input lengths can create CUDA OOM errors. To avoid this,
        we follow `PyTorch notes <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length>`_
        and pre-allocate the memory with a blank forward and backward pass.

    Args:
        batch (Dict[str, Tensor]): The input batch to the model, must be a dictionary.
        curr_seq_length (int): The desired sequence length to apply.
        truncate (bool, optional): Truncate sequences early, or reshape tensors to create
            new examples out of the extra tokens. Default: ``True``.

    Returns:
        Dict[str, Tensor]: a Mapping of input tensors to the model,
            where all tensors have curr_seq_len in the second dimension.

    Example:

    .. code-block::

        import composer.functional as cf

        for epoch in range(num_epochs):
            for X, y in train_loader:
                X = cf.set_batch_sequence_length(X, sequence_length)
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
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

    This algorithm runs on :attr:`~composer.core.event.Event.AFTER_DATALOADER` to modify
    the sequence length of a batch of data, after the model and data have been moved to
    accelerators.

    .. note::

        ``step_size`` should be a `multiple of eight <https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/>`_ for
        optimal throughput on NVIDIA GPUs

    .. note::

        Variable input lengths can create CUDA OOM errors. To avoid this,
        we follow `PyTorch notes <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length>`_
        and pre-allocate the memory with a blank forward and backward pass.

    See the :doc:`Method Card </method_cards/seq_length_warmup>` for more details.

    Example:

    .. code-block::

        from composer.algorithms import SeqLengthWarmup
        from composer import Trainer

        seq_length_warmup = SeqLengthWarmup(duration=0.5,
                                            min_seq_length=8,
                                            max_seq_length=1024,
                                            ste_size=8,
                                            truncate=False)

        trainer = Trainer(model=model,
                          train_dataloader=train_dataloader,
                          max_duration="1ep",
                          algorithms=[seq_length_warmup])

    Args:
        duration (float, optional): Fraction of total training for sequential length
            learning. Default = ``0.3``.
        min_seq_length (int, optional): Minimum sequence length to start the warmup.
            Default = ``8``.
        max_seq_length (int, optional): Maximum sequence length to stop the warmup.
            Default = ``1024``.
        step_size (int, optional): Step size of sequence length. Default = ``8``.
        truncate (bool, optional): Truncate tensors or reshape extra tokens to new
            examples. Default = ``True``.
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
        return (event == Event.INIT and self._original_model is None) or event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        if event == Event.INIT:
            if not isinstance(state.model, ComposerTransformer):
                raise RuntimeError(
                    textwrap.dedent(f"""\
                    {type(self).__name__} requires state.model to be of type {ComposerTransformer.__name__}, not of type {type(state.model)}"""
                                   ))

            if state.train_dataloader.batch_size is None:
                raise RuntimeError("Sequence Length Warmup algorithm requires constant batch size.")

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
                raise RuntimeError("Sequence Length Warmup algorithm requires constant batch size.")
            per_gpu_batch = ceil(per_gpu_macrobatch / state.grad_accum)

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

        if state.max_duration.unit == TimeUnit.EPOCH:
            num_optimization_steps = state.steps_per_epoch * state.max_duration.value
        elif state.max_duration.unit == TimeUnit.BATCH:
            num_optimization_steps = state.max_duration.value
        else:
            raise NotImplementedError(
                textwrap.dedent("""\
                    To use sequential length warmup, the max_duration must be in epochs or batches.
                    Specifying the `max_duration` in tokens or samples for use with sequential
                    length warmup will be supported in a future Composer release. See
                    https://github.com/mosaicml/composer/issues/226."""))
        num_warmup_steps = int(num_optimization_steps * self.duration)  # in batches

        # assume the full sequence length is the unaltered sequence length
        num_update_steps = (self.max_seq_length - self.min_seq_length) // self.step_size
        update_every_n_steps = num_warmup_steps // num_update_steps

        curr_seq_len = self.step_size * (int(state.timer.batch) // update_every_n_steps)
        curr_seq_len = max(curr_seq_len, self.min_seq_length)
        curr_seq_len = min(curr_seq_len, self.max_seq_length)

        state.batch = set_batch_sequence_length(state.batch_dict, curr_seq_len, self.truncate)

        batch_size = state.batch_dict['input_ids'].shape[0]
        logger.metric_batch({
            'seq_length_warmup/curr_seq_len': curr_seq_len,
            'seq_length_warmup/curr_bs': batch_size,
        })
