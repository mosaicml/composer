# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core code for sequence length warmup."""

import textwrap
from math import ceil
from typing import Dict, Mapping, Optional

import torch
import torch.utils.data

from composer.core import Algorithm, Event, State
from composer.core.precision import get_precision_context
from composer.core.time import TimeUnit
from composer.core.types import Batch
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import dist, ensure_tuple

__all__ = ['SeqLengthWarmup', 'set_batch_sequence_length']


def set_batch_sequence_length(
    batch: Dict[str, torch.Tensor],
    curr_seq_len: int,
    preserve_eos: bool = False,
) -> Batch:
    """Set the sequence length of a batch.

    Changes the sequence length of all tensors in the provided dictionary
    to ``curr_seq_len`` by truncating the sequence tensors.

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
        preserve_eos (bool, optional): Preserve the end-of-sequence of the batch when
            truncating. Useful when input formats include a unique end-of-sequence token.
            Default = ``False``.

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
    assert 'attention_mask' in batch

    if curr_seq_len >= batch['attention_mask'].shape[1]:
        return batch

    # Truncate, but preserve end-of-sequence tokens
    if preserve_eos:
        r_idx = torch.arange(batch['attention_mask'].shape[0])
        # eos_idx should point to the final token index for each batch sample
        eos_idx = batch['attention_mask'].sum(1).long() - 1
        # eos_idx_truncated is the same thing, after truncation is applied
        eos_idx_truncated = eos_idx.clamp(max=curr_seq_len - 1)

        for k in batch.keys():
            if len(batch[k].shape) < 2:
                continue
            eos_value = batch[k][r_idx, eos_idx]
            batch[k] = batch[k][:, :curr_seq_len].contiguous()
            batch[k][r_idx, eos_idx_truncated] = eos_value

    else:
        for k in batch.keys():
            if len(batch[k].shape) < 2:
                continue
            batch[k] = batch[k][:, :curr_seq_len].contiguous()

    return batch


class SeqLengthWarmup(Algorithm):
    """Progressively increases the sequence length during training.

    Changes the sequence length of all tensors in the input batch. The
    sequence length increases from ``min_seq_length`` to ``max_seq_length``
    in steps of ``step_size`` during the first ``duration`` fraction of
    training.

    The sequence length is then kept at ``max_seq_length``
    for the rest of training.

    This algorithm runs on :attr:`~composer.core.event.Event.AFTER_DATALOADER` to modify
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
                                            preserve_eos=False)

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
        preserve_eos (bool, optional): Preserve the end-of-sequence of the batch when
            truncating. Useful when input formats include a unique end-of-sequence token.
            Default = ``False``.
    """

    def __init__(
        self,
        duration: float = 0.3,
        min_seq_length: int = 8,
        max_seq_length: int = 1024,
        step_size: int = 8,
        preserve_eos: bool = False,
    ):
        self.duration = duration
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.step_size = step_size
        self.preserve_eos = preserve_eos

        if self.duration < 0 or self.duration > 1:
            raise ValueError(f'Duration must be between 0 and 1, got: {self.duration}')

        if self.max_seq_length < self.min_seq_length:
            raise ValueError(f'max_seq_length={self.max_seq_length} must be '
                             f'greater than min_seq_length={self.min_seq_length}')
        self._activated = False
        self._original_model = None
        self._failed_grad_accums = []
        self._last_seq_len = -1

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

            batch_clone = {}
            device_batch_size = 0
            device = None
            for k, v in state.batch.items():
                # Truncate any sequence-shaped tensors to at most the max_seq_length sequence.
                # Assume the second dimension is always sequence position.
                if len(v.shape) > 1:
                    seq_dim = min(self.max_seq_length, v.shape[1])
                    batch_clone[k] = torch.clone(v[:, :seq_dim])
                else:
                    batch_clone[k] = torch.clone(v)
                device_batch_size = v.shape[0]
                device = v.device

            grad_accum_successful = False
            while not grad_accum_successful:
                print(f'Trying pre-activation for SLW with grad_accum={state.grad_accum} ... ')
                per_gpu_batch = ceil(per_gpu_macrobatch / state.grad_accum)
                model_inputs = {k: v[:per_gpu_batch] for k, v in batch_clone.items()}

                should_handle_cuda_oom = 0
                caught_timeout_error = None
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

                    for optimizer in state.optimizers:
                        optimizer.zero_grad()

                # This error/state.grad_accum handling mimics the logic in trainer._train_batch().
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        should_handle_cuda_oom = 1
                    elif 'Timed out' in str(e):
                        # Catch timeout errors and only reraise if we did not encounter OOM on other ranks. Error
                        # is likely transient if one rank OOMed, it likely did not reach a barrier. Note that if we
                        # catch non-transient timeout errors they will be later reraised if no rank OOMed.
                        caught_timeout_error = e
                    else:
                        raise

                # Propagate across all ranks if any rank hit CUDA OOM
                should_handle_cuda_oom = torch.tensor([should_handle_cuda_oom], dtype=torch.uint8, device=device)
                dist.all_reduce(should_handle_cuda_oom, reduce_operation='MAX')
                if int(should_handle_cuda_oom.item()) == 1:
                    # If any rank hit CUDA OOM, update grad_accum and retry. Ignore any caught_timeout_error since
                    # it is likely transient, e.g. timeout because certain ranks OOMed and didn't reach barrier.
                    # Raise runtime error if training 1 sample at a time still resulted in CUDA out of memory
                    if state.grad_accum == device_batch_size:
                        raise RuntimeError(
                            ('CUDA out of memory. The train loop failed with an internal microbatch of size 1.'
                             'The GPU does not have enough memory to process even 1 sample.'))
                    else:
                        state.grad_accum = min(2 * state.grad_accum, device_batch_size)
                        logger.data_batch({'trainer/grad_accum': state.grad_accum})
                elif caught_timeout_error:
                    # If not CUDA out of memory, raise exception to user. Note that this truncates the call stack
                    # back only to this newly raised error.
                    raise caught_timeout_error
                else:
                    grad_accum_successful = True

                print(f"{'Success!' if grad_accum_successful else 'Failure...'}\n")

            self._activated = True

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

        print('num optimization steps', num_optimization_steps)
        print('num_warmup_steps', num_warmup_steps)
        print('num_update_steps', num_update_steps)
        print('update_every_n_steps', update_every_n_steps)
        print('timestamp.batch', int(state.timestamp.batch))

        curr_seq_len = self.step_size * (int(state.timestamp.batch) // update_every_n_steps) + self.min_seq_length
        print('curr_seq_len', curr_seq_len)
        curr_seq_len = max(curr_seq_len, self.min_seq_length)
        curr_seq_len = min(curr_seq_len, self.max_seq_length)

        if curr_seq_len != self._last_seq_len:
            print(f'At batch {int(state.timestamp.batch)}, current sequence length = {curr_seq_len}.')
        self._last_seq_len = int(curr_seq_len)

        state.batch = set_batch_sequence_length(state.batch, curr_seq_len, self.preserve_eos)

        batch_size = state.batch['input_ids'].shape[0]
        logger.data_batch({
            'seq_length_warmup/curr_seq_len': curr_seq_len,
            'seq_length_warmup/curr_bs': batch_size,
        })
