# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running distributed data parallel training."""

import datetime
import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, ContextManager, Union, cast

import torch
import torch.distributed as torch_dist
import torch.nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel

from composer.core.state import State
from composer.utils import dist
from composer.utils.string_enum import StringEnum

__all__ = ['DDPSyncStrategy', 'ddp_sync_context', 'prepare_ddp_module']

log = logging.getLogger(__name__)


class DDPSyncStrategy(StringEnum):
    """How and when DDP gradient synchronization should happen.

    Attributes:
        SINGLE_AUTO_SYNC: The default behavior for DDP. Gradients are synchronized as they
            computed, for only the final microbatch of a batch. This is the most efficient
            strategy, but can lead to errors when ``find_unused_parameters`` is set, since
            it is possible different microbatches may use different sets of parameters,
            leading to an incomplete sync.
        MULTI_AUTO_SYNC: The default behavior for DDP when ``find_unused_parameters`` is set.
            Gradients are synchronized as they are computed for all microbatches. This ensures
            complete synchronization, but is less efficient than :attr:`SINGLE_AUTO_SYNC`. This
            efficiency gap is usually small, as long as either DDP syncs are a small portion
            of the trainer's overall runtime, or the number of microbatches per batch is
            relatively small.
        FORCED_SYNC: Gradients are manually synchronized only after all gradients have been
            computed for the final microbatch of a batch. Like :attr:`MULTI_AUTO_SYNC`, this
            strategy ensures complete gradient synchronization, but this tends to be slower than
            :attr:`MULTI_AUTO_SYNC`. This is because ordinarily syncs can happen in parallel
            with the ``loss.backward()`` computation, meaning syncs can be mostly complete by
            the time that function finishes. However, in certain circumstances, syncs may take
            a very long time to complete - if there are also a lot of microbatches per batch,
            this strategy may be optimal.
    """
    SINGLE_AUTO_SYNC = 'single_auto_sync'
    MULTI_AUTO_SYNC = 'multi_auto_sync'
    FORCED_SYNC = 'forced_sync'


@contextmanager
def ddp_sync_context(state: State, is_final_microbatch: bool, sync_strategy: Union[str, DDPSyncStrategy]):
    """A context manager for handling the :class:`DDPSyncStrategy`.

    Args:
        state (State): The state of the :class:`~composer.trainer.trainer.Trainer`.
        is_final_microbatch (bool): Whether or not the context is being used during the final
            microbatch of the gradient accumulation steps.
        sync_strategy (str | DDPSyncStrategy): The ddp sync strategy to use. If a string
            is provided, the string must be one of the values in :class:`DDPSyncStrategy`.
    """
    if not isinstance(state.model, DistributedDataParallel):
        yield
        return

    assert state.optimizers is not None, 'optimizers have not been initialized'
    sync_strategy = DDPSyncStrategy(sync_strategy)

    no_sync_context = cast(Callable[[], ContextManager], state.model.no_sync)
    auto_sync_context = nullcontext

    if sync_strategy == DDPSyncStrategy.SINGLE_AUTO_SYNC:
        context = auto_sync_context if is_final_microbatch else no_sync_context
        with context():
            yield

    elif sync_strategy == DDPSyncStrategy.MULTI_AUTO_SYNC:
        with auto_sync_context():
            yield

    elif sync_strategy == DDPSyncStrategy.FORCED_SYNC:
        try:
            with no_sync_context():
                yield
        finally:
            if is_final_microbatch:
                for optimizer in state.optimizers:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                dist.all_reduce(p.grad)
                                p.grad = p.grad / dist.get_world_size()

    else:
        raise ValueError('Unknown sync strategy', sync_strategy)


def prepare_ddp_module(module: torch.nn.Module, find_unused_parameters: bool,
                       adaptive_gradient_accumulation: bool) -> torch.nn.Module:
    """Wraps the module in a :class:`torch.nn.parallel.DistributedDataParallel` object if running distributed training.

    Args:
        module (torch.nn.Module): The module to wrap.
        find_unused_parameters (bool): Whether or not to do a pass over the autograd graph
            to find parameters to not expect gradients for. This is useful if there are some
            parameters in the model that are not being trained.
        adaptive_gradient_accumulation (bool): Whether adaptive gradient accumulation is enabled.
            If enabled, we require inserting a barrier ahead of gradient reduction to avoid deadlock.
    """
    if dist.is_available() and dist.is_initialized():
        if any((p.requires_grad for p in module.parameters())):
            log.debug('Wrapping model with DistributedDataParallel')
            ddp_model = DistributedDataParallel(module, find_unused_parameters=find_unused_parameters)
            if adaptive_gradient_accumulation:
                # Wrap the default reduce hook with a barrier
                ddp_model.register_comm_hook(torch_dist.get_backend(), rank_sync_wrapper(allreduce_hook))
            return ddp_model
        return module
    if dist.is_available():
        raise RuntimeError('Please call dist.initialize_dist() before calling ddp.prepare_module()')

    raise RuntimeError('When the world size is > 1, ``torch.distributed`` must be used. However, it is '
                       'not available in your installation of PyTorch. Please install or build PyTorch '
                       'with distributed support.')


def rank_sync_wrapper(
    hook: Callable[[Any, torch_dist.GradBucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, torch_dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """Wrapper to insert monitored_barrier if using adaptive gradient accumulation.

    If a subset of ranks OOM, this monitored barrier fails and the error is caught so training can continue.
    Otherwise, two ranks would enter different barriers, resulting in deadlock.
    """

    def rank_sync_wrapper_hook(hook_state, bucket: torch_dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        try:
            dist.monitored_barrier(timeout=datetime.timedelta(seconds=30))
        except RuntimeError as e:
            # monitored_barrier was tripped
            if 'Timed out' in str(e):

                def raise_timeout_error():
                    raise e

                # Use a no-op hook and return the same gradients already on the device. If we don't
                # do the reduction, PyTorch will raise an internal error on the next backward pass
                # as the previous reduction hasn't been completed. After completing the no-op
                # reduction, re-raise the timeout error.
                fut = torch.futures.Future()
                fut.set_result(bucket.get_tensors())
                return fut.then(raise_timeout_error)
            else:
                raise
        return hook(hook_state, bucket)

    return rank_sync_wrapper_hook
