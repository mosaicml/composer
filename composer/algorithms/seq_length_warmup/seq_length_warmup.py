# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core code for sequence length warmup."""

import logging
import textwrap
from math import ceil
from typing import Dict, Mapping, Optional

import torch
import torch.utils.data

from composer.core import Algorithm, Batch, Event, State, TimeUnit, get_precision_context
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import dist, ensure_tuple

log = logging.getLogger(__name__)

__all__ = ['SeqLengthWarmup', 'set_batch_sequence_length']


def set_batch_sequence_length(
    batch: Dict[str, torch.Tensor],
    curr_seq_len: int,
    truncate: bool = True,
    preserve_end_of_sequence: bool = False,
) -> Batch:
    """Set the sequence length of a batch.

    Changes the sequence length of all tensors in the provided dictionary
    to ``curr_seq_len`` by either truncating the tensors (``truncate=True``)
    or reshaping the tensors to create new examples from the extra tokens
    (``truncate=False``).

    .. note::

        The schedule for ``curr_seq_len`` over training time should be managed
        outside of this function.

    .. note::

        Variable input lengths can create CUDA OOM errors. To avoid this,
        we follow the `PyTorch notes <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length>`_
        and pre-allocate the memory with a blank forward and backward pass.

    Args:
        batch (Dict[str, Tensor]): The input batch to the model, must be a dictionary.
        curr_seq_length (int): The desired sequence length to apply.
        truncate (bool, optional): Truncate sequences early, or reshape tensors to create
            new examples out of the extra tokens. Default: ``True``.
        preserve_end_of_sequence (bool, optional): Preserve the end-of-sequence of the batch when
            truncating. Useful when input formats include a unique end-of-sequence token.
            Ignored if ``truncate=False``. Default: ``False``.
            E.g., if ``batch["input_ids"]`` is ``[[10, 11, 12, 13, 14, 15]]``
            and ``curr_seq_length=3``, ``"input_ids"`` in the returned batch would be
            ``[[10, 11, 12]]`` with ``preserve_end_of_sequence=False`` and would be
            ``[[10, 11, 15]]`` with ``preserve_end_of_sequence=True``. This behavior applies to any
            batch tensor with 2 or more dimensions.

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

    # This should act like a no-op if curr_seq_len isn't shorter than the batch's sequence length
    batch_seq_len = 0
    # The batch sequence length is assumed to be the shape[1] dimension of any non-1D batch tensor
    for batch_tensor in batch.values():
        tensor_shape = batch_tensor.shape
        if len(tensor_shape) > 1:
            batch_seq_len = tensor_shape[1]
            break
    if curr_seq_len >= batch_seq_len:
        return batch

    if truncate:
        # Truncate, but preserve end-of-sequence tokens
        if preserve_end_of_sequence:
            if 'attention_mask' not in batch:
                raise ValueError(
                    'Sequence Length Warmup requires that the batch has "attention_mask" when using ``preserve_end_of_sequence=True``.'
                )
            r_idx = torch.arange(batch['attention_mask'].shape[0])
            # eos_idx should point to the final token index for each batch sample
            eos_idx = batch['attention_mask'].sum(dim=1).long() - 1
            # eos_idx_truncated is the same thing, after truncation is applied
            eos_idx_truncated = eos_idx.clamp(max=curr_seq_len - 1)

            for k in batch.keys():
                if batch[k].ndim < 2:
                    raise ValueError(
                        f'Sequence Length Warmup requires that all tensors are sequence-shaped when ``truncate=True``. '
                        f'Tensor "{k}" has shape {batch[k].shape}.')
                eos_value = batch[k][r_idx, eos_idx]
                batch[k] = batch[k][:, :curr_seq_len].contiguous()
                batch[k][r_idx, eos_idx_truncated] = eos_value

        else:
            for k in batch.keys():
                if batch[k].ndim < 2:
                    raise ValueError(
                        f'Sequence Length Warmup requires that all tensors are sequence-shaped when ``truncate=True``. '
                        f'Tensor "{k}" has shape {batch[k].shape}.')
                batch[k] = batch[k][:, :curr_seq_len].contiguous()

    else:
        if 'input_ids' not in batch:
            raise ValueError(
                'Sequence Length Warmup requires that the batch has "input_ids" when using ``truncate=False``.')
        input_ids_shape = batch['input_ids'].shape
        # ensure new tensor shape is divisible by curr_seq_len
        input_ids = batch['input_ids'].view(-1)
        tensor_len = (input_ids.shape[0] // curr_seq_len) * curr_seq_len

        input_ids = input_ids[:tensor_len]
        input_ids = input_ids.view(-1, curr_seq_len)
        batch['input_ids'] = input_ids

        for k, v in batch.items():
            if k == 'input_ids':
                continue
            if v.shape != input_ids_shape:
                raise ValueError(
                    f'When using ``truncate=False``, Sequence Length Warmup only supports batches where all tensors have the same shape. '
                    f'Tensor "{k}" has shape {v.shape} but should have shape {input_ids_shape}.')
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

    This algorithm runs on :attr:`.Event.AFTER_DATALOADER` to modify
    the sequence length of a batch of data after the model and data have been moved to
    accelerators.

    .. note::

        ``step_size`` should be a `multiple of eight <https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/>`_ for
        optimal throughput on NVIDIA GPUs.

    .. note::

        Variable input lengths can create CUDA OOM errors. To avoid this,
        we follow the `PyTorch notes <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length>`_
        and pre-allocate the memory with a blank forward and backward pass.

    See the :doc:`Method Card </method_cards/seq_length_warmup>` for more details.

    Example:

    .. code-block::

        from composer.algorithms import SeqLengthWarmup
        from composer import Trainer

        seq_length_warmup = SeqLengthWarmup(duration=0.5,
                                            min_seq_length=8,
                                            max_seq_length=1024,
                                            step_size=8,
                                            truncate=True,
                                            preserve_end_of_sequence=False)

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
        truncate (bool, optional): Truncate sequences early, or reshape tensors to create
            new examples out of the extra tokens. Default: ``True``.
        preserve_end_of_sequence (bool, optional): Preserve the end-of-sequence of the batch when
            truncating. Useful when input formats include a unique end-of-sequence token.
            Ignored if ``truncate=False``. Default: ``False``.
            E.g., if ``batch["input_ids"]`` is ``[[10, 11, 12, 13, 14, 15]]``
            and ``curr_seq_length=3``, ``"input_ids"`` in the returned batch would be
            ``[[10, 11, 12]]`` with ``preserve_end_of_sequence=False`` and would be
            ``[[10, 11, 15]]`` with ``preserve_end_of_sequence=True``. This behavior applies to any
            batch tensor with 2 or more dimensions.
    """

    def __init__(
        self,
        duration: float = 0.3,
        min_seq_length: int = 8,
        max_seq_length: int = 1024,
        step_size: int = 8,
        truncate: bool = True,
        preserve_end_of_sequence: bool = False,
    ):
        self.duration = duration
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.step_size = step_size
        self.truncate = truncate
        self.preserve_end_of_sequence = preserve_end_of_sequence

        if self.duration < 0 or self.duration > 1:
            raise ValueError(f'Duration must be between 0 and 1, got: {self.duration}')

        if self.max_seq_length < self.min_seq_length:
            raise ValueError(f'max_seq_length={self.max_seq_length} must be '
                             f'greater than min_seq_length={self.min_seq_length}')
        self._activated = False
        self._original_model = None

    def _activate_model(self, state: State, logger: Logger) -> None:
        """Does a forward and a backward pass on a dummy input.

        The purpose of activating the model is to prevent OOMs. This happens two ways.

        First, this prevents GPU memory from being reallocated when the sequence
        length increases.

        Second, it detects if the batch*max_sequence_length size will cause an OOM and
        increases state.grad_accum accordingly. This logic mirrors the ``grad_accum="auto"``
        logic in :class:`.Trainer`.
        """

        assert self._original_model is not None, 'original model should be set on Event.INIT'

        try:
            # Both PyTorch and FFCV dataloaders define a `batch_size` attribute
            # This exception would mainly be raised if the user is passing in a custom
            # iterable
            per_gpu_macrobatch = getattr(state.dataloader, 'batch_size')
        except AttributeError as e:
            raise AttributeError(
                'Sequence Length Warmup requires the `state.dataloader` to have a `batch_size` attribute.') from e
        if per_gpu_macrobatch is None:
            raise RuntimeError('Sequence Length Warmup algorithm requires constant batch size.')

        # truncate all sequence-shaped tensors to the max sequence length
        batch_clone = {k: torch.clone(v) for k, v in state.batch.items()}
        device_batch_size = 0
        for k, v in batch_clone.items():
            if v.ndim < 2:
                raise ValueError(f'Sequence Length Warmup requires that all tensors are sequence-shaped. '
                                 f'Tensor "{k}" has shape {v.shape}.')
            batch_clone[k] = v[:, :self.max_seq_length].contiguous()
            device_batch_size = v.shape[0]

        # In-line to avoid circular dependency
        from composer.trainer.trainer import _adjust_device_train_microbatch_size, _adjust_grad_accum, _is_cuda_oom

        # This loop tries to do a forward/backward pass using the current microbatch size.
        # If it hits an OOM error, it doubles `state.grad_accum` and tries again until
        # it succeeds.
        while True:
            per_gpu_batch = ceil(per_gpu_macrobatch / state.grad_accum)
            model_inputs = {k: v[:per_gpu_batch] for k, v in batch_clone.items()}

            found_cuda_oom = 0  # int since bool BOR not supported on all torch.distributed backends
            try:
                # start by running a forward and backward pass
                # of the maximum sequence length to allocate cache.
                with get_precision_context(state.precision):
                    outputs = state.model.forward(model_inputs)
                    loss = self._original_model.loss(outputs, model_inputs)

                # since use_grad_scaling is in the Trainer, and we
                # don't care about the loss values, skip scaling
                for loss_item in ensure_tuple(loss):
                    loss_item.backward()

                # Zero any gradients created by the backward pass
                for optimizer in state.optimizers:
                    optimizer.zero_grad()

            # This error/state.grad_accum handling mimics the logic in trainer._train_batch().
            except RuntimeError as e:
                if state.auto_microbatching and _is_cuda_oom(e):
                    log.debug((f"Rank {dist.get_global_rank()} OOM'd."))
                    found_cuda_oom = 1
                else:
                    raise

            if state.auto_microbatching:
                # Propagate across all ranks if any rank hit CUDA OOM
                found_cuda_oom = state.device.tensor_to_device(torch.tensor([found_cuda_oom], dtype=torch.uint8))
                dist.all_reduce(found_cuda_oom, reduce_operation='MAX')
                if found_cuda_oom.item() == 1:
                    if state.using_device_microbatch_size:
                        _adjust_device_train_microbatch_size(state)
                    else:
                        _adjust_grad_accum(state, device_batch_size)
                    # Skip return and rerun after handling oom
                    continue
            # Activate and return if we've completed without OOMing.
            self._activated = True
            return

    def match(self, event: Event, state: State) -> bool:
        return (event == Event.INIT and self._original_model is None) or event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        if event == Event.INIT:
            if not isinstance(state.model, HuggingFaceModel):
                raise RuntimeError(
                    textwrap.dedent(f"""\
                    {type(self).__name__} requires state.model to be of type {HuggingFaceModel.__name__}, not of type {type(state.model)}"""
                                   ))

            self._original_model = state.model
            return

        assert state.dataloader is not None, 'dataloader should be set on AFTER_DATALOADER'
        assert state.max_duration is not None, 'max_duration should be set on AFTER_DATALOADER'

        # in order to avoid OOMs, we do a forward and a backward pass on a dummy input.
        if not self._activated:
            self._activate_model(state, logger)

        if state.max_duration.unit == TimeUnit.EPOCH:
            if state.dataloader_len is None:
                raise RuntimeError('Sequential Length Warmup requires the dataloader to be sized.')
            num_optimization_steps = int(state.dataloader_len) * state.max_duration.value
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

        curr_seq_len = self.step_size * (int(state.timestamp.batch) // update_every_n_steps) + self.min_seq_length
        curr_seq_len = max(curr_seq_len, self.min_seq_length)
        curr_seq_len = min(curr_seq_len, self.max_seq_length)

        state.batch = set_batch_sequence_length(state.batch, curr_seq_len, self.truncate, self.preserve_end_of_sequence)

        batch_size = state.batch['input_ids'].shape[0]
        logger.log_metrics({
            'seq_length_warmup/curr_seq_len': curr_seq_len,
            'seq_length_warmup/curr_bs': batch_size,
        })
