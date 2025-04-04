# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running distributed data parallel training."""

import logging
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, ContextManager, Iterator, Optional, Sequence, Union, cast

import torch
from packaging import version
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric, MetricCollection

from composer.core import Precision, State
from composer.core.precision import _validate_precision
from composer.devices import Device, DeviceGPU
from composer.distributed.mosaic_parallelism import (
    BACKWARD_PREFETCH_MAP,
    SHARDING_MAP,
    get_cpu_offload,
    get_mixed_precision,
    set_custom_fsdp_module_kwargs,
)
from composer.utils import FSDPConfig, StringEnum, TPConfig, dist, ensure_tuple, get_device

__all__ = ['DDPSyncStrategy', 'ddp_sync_context', 'prepare_ddp_module', 'prepare_fsdp_module', 'prepare_tp_module']

log = logging.getLogger(__name__)

process_group_cache = {}


class DDPSyncStrategy(StringEnum):
    """How and when gradient synchronization should happen.

    Attributes:
        SINGLE_AUTO_SYNC: The default behavior. Gradients are synchronized as they
            computed, for only the final microbatch of a batch. This is the most efficient
            strategy, but can lead to errors when ``find_unused_parameters`` is set, since
            it is possible different microbatches may use different sets of parameters,
            leading to an incomplete sync.
        MULTI_AUTO_SYNC: The default behavior when ``find_unused_parameters`` is set.
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
        state (State): The state of the :class:`.Trainer`.
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


def prepare_ddp_module(module: torch.nn.Module, find_unused_parameters: bool) -> torch.nn.Module:
    """Wraps the module in a :class:`torch.nn.parallel.DistributedDataParallel` object if running distributed training.

    Args:
        module (torch.nn.Module): The module to wrap.
        find_unused_parameters (bool): Whether or not to do a pass over the autograd graph
            to find parameters to not expect gradients for. This is useful if there are some
            parameters in the model that are not being trained.
    """
    if dist.is_available() and dist.is_initialized():
        if any((p.requires_grad for p in module.parameters())):
            log.debug('Wrapping model with DistributedDataParallel')
            ddp_model = DistributedDataParallel(module, find_unused_parameters=find_unused_parameters)
            return ddp_model
        return module
    if dist.is_available():
        raise RuntimeError('Please call dist.initialize_dist() before calling ddp.prepare_module()')

    raise RuntimeError(
        'When the world size is > 1, ``torch.distributed`` must be used. However, it is '
        'not available in your installation of PyTorch. Please install or build PyTorch '
        'with distributed support.',
    )


def _recreate_fsdp_param_groups_from_unwrapped_opt_info(
    fsdp_wrapped_named_params: Iterator[tuple[str, torch.nn.Parameter]],
    non_wrapped_param_names_to_group_num: dict[str, int],
    group_num_to_optimizer_info: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Helper function to recreate optimizer groups for FSDP wrapped modules.

    Optimizer param groups are formatted as:
    [
        {'params': [p1, p2], 'lr' : lr1}, # group 0
        {'params': [p3], 'lr' : lr2} # group 1
    ]
    ie. there are multiple parameters per group. Here, we track the group number in order to map
    multiple parameters to the same group

    Args:
        fsdp_wrapped_named_params: output of model.named_parameters() of an FSDP wrapped model
        non_wrapped_param_names_to_group_num: a dict mapping from the original model param names
                                            to the param group number
        group_num_to_optimizer_info: stores info like lr, eps for each group

    Returns a list of param groups, referencing the fsdp parameters
    """
    # Initialize an empty list of parameters for each optimizer group
    for group_num in group_num_to_optimizer_info.keys():
        group_num_to_optimizer_info[group_num]['params'] = []

    for fsdp_name, param in fsdp_wrapped_named_params:

        unwrapped_name = clean_tensor_name(fsdp_name)

        # Since we are iterating over all model.named_parameters() after fsdp wrapping, we need to check
        # if the parameter was included in the optimizer param_group pre fsdp wrapping, in order to support
        # passing a subset of model params in the optimizer
        if unwrapped_name in non_wrapped_param_names_to_group_num:
            # Need to have a 1:1 mapping between a fsdp param name and the non-wrapped vanilla param name
            retrieved_group_num = non_wrapped_param_names_to_group_num[unwrapped_name]
            group_num_to_optimizer_info[retrieved_group_num]['params'].append(param)

    # return sorted optimizer info groups
    return [group_num_to_optimizer_info[num] for num in sorted(group_num_to_optimizer_info.keys())]


def prepare_tp_module(
    model: torch.nn.Module,
    optimizers: Optional[Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]],
    tp_config: TPConfig,
) -> None:
    """Prepare a module (assumed ComposerModel) for use with tensor parallel."""
    optimizers_tuple = ensure_tuple(optimizers)
    if len(optimizers_tuple) != 1:
        raise NotImplementedError(f'Only one optimizer is supported; found {len(optimizers_tuple)} optimizers')

    optim = optimizers_tuple[0]
    if len(optim.param_groups) > 1:
        raise RuntimeError('Multiple optimizer groups are not supported with tensor parallelism.',)

    if len(optim.param_groups[0]['params']) != len(list(model.parameters())):
        raise ValueError(
            'Passing in a subset of model parameters to the optimizer is not supported with tensor parallelism.',
        )

    from torch.distributed.tensor.parallel import parallelize_module

    device_mesh = tp_config.device_mesh
    assert device_mesh is not None  # For type checking, set in State.__init__
    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=tp_config.layer_plan,
    )


def prepare_fsdp_module(
    model: torch.nn.Module,
    optimizers: Optional[Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]],
    fsdp_config: FSDPConfig,
    precision: Optional[Union[str, Precision]] = None,
    device: Optional[Union[str, Device]] = None,
    auto_microbatching: bool = False,
    te_rng_seed: int = 1234,
) -> tuple[list, dict]:
    """Prepare a module (assumed ComposerModel) and optimizer for use with :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    Args:
        model (torch.nn.Module): The model to wrap.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): The optimizer for `model`, assumed to have a single param group := model.parameters().
        fsdp_config (FSDPConfig): The FSDP config.
        precision: (Precision): The precision being used by the Trainer, used to fill in defaults for FSDP `mixed_precision` settings.
        device:  The device being used by the Trainer.
        auto_microbatching (bool, optional): Whether or not auto microbatching is enabled.
        te_rng_seed(int): The seed to use for the Transformer Engine activation checkpointing RNG. Defaults to 1234.
    """
    device = get_device(device)

    if precision is None:
        precision = Precision.AMP_FP16 if isinstance(device, DeviceGPU) else Precision.FP32
    elif isinstance(precision, str):
        precision = Precision(precision)
    _validate_precision(precision, device)

    # Check sync_module_states is True for mixed initialization or HSDP
    if fsdp_config.sync_module_states == False:
        rank_on_meta = 1 if next(model.parameters()).device.type == 'meta' else 0
        all_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
        dist.all_reduce(all_ranks_meta, reduce_operation='MIN')
        any_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
        dist.all_reduce(any_ranks_meta, reduce_operation='MAX')
        if all_ranks_meta.item() == 0 and any_ranks_meta.item() == 1:
            raise ValueError(
                'Detected mixed initialization where some ranks have model on cpu or '
                'gpu and some ranks are on meta. Either keep all ranks on the same '
                "device or set parallelism_config['fsdp']['sync_module_states'] = True. Otherwise, "
                'some weights may be randomly initialized when loading a checkpoint.',
            )

    # Handles of FSDP sync hooks if automicrobatching is on
    hook_handles = []

    # Check if other ranks OOMed after forward/backward pass when using auto microbatching. This
    # may happen when close to memory limit or with uneven memory usage across ranks. Since we
    # need to do this before the model weights are gathered for the next FSDP block, we wrap every
    # FSPD block with a hook that checks if any other rank OOMed.
    def sync_hook(*args):
        # Check if any other rank hit an OOM
        found_cuda_oom_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
        found_cuda_oom = found_cuda_oom_tensor.item()
        # Signal current rank is still in batch
        all_ranks_finished_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')

        if found_cuda_oom == 1:
            raise RuntimeError('CUDA out of memory encountered on a different rank')

    # Necessary variables for optimizers with multiple param groups in FSDP
    param_name_to_group_num = None
    group_num_to_opt_group_info = None
    single_param_group_opt_info = None

    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        if len(optimizers_tuple) != 1:
            raise NotImplementedError(f'Only one optimizer is supported; found {len(optimizers_tuple)} optimizers')

        # clearing optimizer param groups and state
        # that will be recreated at the end of prepare_fsdp_module
        optim = optimizers_tuple[0]

        # Simplest case - single param group & all model params stored in optimizer
        if len(optim.param_groups) == 1 and len(optim.param_groups[0]['params']) == len(list(model.parameters())):
            single_param_group_opt_info = {k: v for k, v in optim.param_groups[0].items() if k != 'params'}
        elif fsdp_config.use_orig_params:
            # this code block stores information about param groups pre-fsdp wrapping in order to recreate them post-wrapping
            # to do so, it relies on the ptrs of the model.parameters() in a model and the names of the params
            # for this to work, use_orig_params=True, as we need the names of the params post-wrapping
            # TP is not supported, as the underlying parameters in the model differ from the params in the param groups after being dtensorified

            ptr_to_param_name = {id(p): n for n, p in model.named_parameters()}
            param_name_to_group_num = {}
            group_num_to_opt_group_info = {}
            for group_num in range(len(optim.param_groups)):
                # Need to in-line to avoid a reference which causes FSDP to allocate extra GPU memory
                # group = optim.param_groups[group_num]
                for param_num in range(len(optim.param_groups[group_num]['params'])):
                    param_ptr = id(optim.param_groups[group_num]['params'][param_num])
                    if param_ptr not in ptr_to_param_name:
                        raise ValueError('The same model must be passed to the optimizer and trainer.')
                    param_name_to_group_num[ptr_to_param_name[param_ptr]] = group_num

                # this includes optimizer-specific values like lr, eps
                # this will be used as the kwargs for the optim param groups later
                optimizer_specific_group_info = {
                    k: v for k, v in optim.param_groups[group_num].items() if k != 'params'
                }
                group_num_to_opt_group_info[group_num] = optimizer_specific_group_info
        else:
            if len(optim.param_groups) > 1:
                raise RuntimeError('Multiple optimizer groups with FSDP are not supported with use_orig_params=False.',)

            if len(optim.param_groups[0]['params']) != len(list(model.parameters())):
                raise ValueError(
                    'Passing in a subset of model parameters to the optimizer is not supported with use_orig_params=False.',
                )

        optim.param_groups.clear()
        optim.state.clear()

    sharding_map_key = fsdp_config.sharding_strategy.upper()
    sharding_strategy = SHARDING_MAP[sharding_map_key]

    kwargs = {}
    if version.parse(
        torch.__version__.split('.dev')[0],
    ) >= version.parse('2.2.0') and fsdp_config.device_mesh is not None:
        if fsdp_config.process_group is not None:
            warnings.warn(
                'process_group and device_mesh are set for FSDP, so ignoring device_mesh. Please set process_group to None.',
            )
        else:
            ndim = fsdp_config.device_mesh.ndim
            if ndim == 1 and sharding_strategy == ShardingStrategy.HYBRID_SHARD:
                sharding_strategy = ShardingStrategy.FULL_SHARD
                warnings.warn('HYBRID_SHARD is not supported with 1D device mesh. Using FULL_SHARD instead.')
            elif ndim == 1 and sharding_strategy == ShardingStrategy._HYBRID_SHARD_ZERO2:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
                warnings.warn('_HYBRID_SHARD_ZERO2 is not supported with 1D device mesh. Using SHARD_GRAD_OP instead.')
            elif ndim == 2 and sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
                warnings.warn('SHARD_GRAD_OP is not supported with 2D device mesh. Using _HYBRID_SHARD_ZERO2 instead.')
            elif ndim == 2 and sharding_strategy == ShardingStrategy.FULL_SHARD:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
                warnings.warn('FULL_SHARD is not supported with 2D device mesh. Using HYBRID_SHARD instead.')
            kwargs['device_mesh'] = fsdp_config.device_mesh

    cpu_offload = get_cpu_offload(cpu_offload=fsdp_config.cpu_offload)

    mixed_precision = fsdp_config.mixed_precision
    keep_low_precision_grads = fsdp_config.keep_low_precision_grads
    mixed_precision, _, _, _ = get_mixed_precision(
        precision,
        mixed_precision=mixed_precision,
        keep_low_precision_grads=keep_low_precision_grads,
    )

    process_group = None
    if fsdp_config.process_group is not None:
        process_group_dict = {'process_group': fsdp_config.process_group}
        process_group = set_custom_fsdp_module_kwargs(process_group_dict, process_group_cache)['process_group']
    backward_prefetch = BACKWARD_PREFETCH_MAP[fsdp_config.backward_prefetch.upper()]
    activation_checkpointing = fsdp_config.activation_checkpointing
    activation_cpu_offload = fsdp_config.activation_cpu_offload
    sync_module_states = fsdp_config.sync_module_states
    forward_prefetch = fsdp_config.forward_prefetch
    limit_all_gathers = fsdp_config.limit_all_gathers
    ignored_modules = fsdp_config.ignored_modules
    state_dict_type = fsdp_config.state_dict_type
    activation_checkpointing_reentrant = fsdp_config.activation_checkpointing_reentrant
    te_checkpoint_wrapper = fsdp_config.te_checkpoint_wrapper if precision == Precision.AMP_FP8 else False
    te_shard_fp8_weight = fsdp_config.te_shard_fp8_weight if precision == Precision.AMP_FP8 else False
    sharded_ckpt_prefix_dir = fsdp_config.sharded_ckpt_prefix_dir
    use_orig_params = fsdp_config.use_orig_params

    fsdp_obj_named_modules = {}
    # We choose to not wrap the ComposerModel directly, but instead wrap any submodules like `ComposerModel.model`
    # This makes it safer to call ComposerModel-specific functions like 'eval_forward' that
    # may make calls to sharded submodules. If we only wrap the submodules, then any call that ComposerModel makes
    # to a FSDP-wrapped submodule's `forward()` function will be safe and all-gather the necessary weights before `forward()`.
    for obj_name, obj in model.named_children():
        if not isinstance(obj, (Metric, MetricCollection)):

            # Skip wrapping submodules which are explicitly marked with no wrap
            if hasattr(obj, '_fsdp_wrap') and not bool(obj._fsdp_wrap):
                continue

            # A dictionary of all tied parameter pointers to (module, attr) tuples
            tied_pointers = {}

            # Goes through all modules finding which weights have the same pointers
            for mod in obj.modules():
                for attr_name, attr in mod.named_parameters(recurse=False):
                    ptr = id(attr)
                    mod_attr_list = tied_pointers.get(ptr, [])
                    mod_attr_list.append((mod, attr_name))
                    tied_pointers[ptr] = mod_attr_list

            # Dictionary mapping the source module to a list of (target module, source attr, target attr) tuples
            source_mod_to_mod_attr = {}
            for mod_attr_list in tied_pointers.values():
                # If there is only one module for this pointer, then there is no weight tying
                if len(mod_attr_list) == 1:
                    continue

                # Arbitrarily choose the first module as the source module
                first_mod, first_attr = mod_attr_list[0]
                source_mod_to_mod_attr[first_mod] = [
                    (target_mod, first_attr, dest_attr) for target_mod, dest_attr in mod_attr_list[1:]
                ]

            # Clean up no longer needed module references for memory safety
            del tied_pointers

            def _param_init_fn(module: torch.nn.Module) -> None:
                # If we do not have any parameters or buffers on meta device managed by this module directly, we do not need to call the parameter init function.
                # It is assumed that whatever process moved the parameters off of meta device initialized them.
                # We expect this to occur if we have tied weights, as the second module will already have the weights initialized.
                is_meta = any(param.is_meta for param in module.parameters(recurse=False)
                             ) or any(buffer.is_meta for buffer in module.buffers(recurse=False))
                if not is_meta:
                    return

                # Move all parameters and buffers to the current device
                module.to_empty(device=f'cuda:{torch.cuda.current_device()}', recurse=False)

                # Redo weight tying, which will have been broken by the above line that moves parameters off of meta device
                if module in source_mod_to_mod_attr:
                    for target_mod, first_attr, dest_attr in source_mod_to_mod_attr[module]:
                        setattr(target_mod, dest_attr, getattr(module, first_attr))

                # Run the specified initialization
                if hasattr(obj, 'param_init_fn') and isinstance(obj.param_init_fn, Callable):
                    obj.param_init_fn(module)
                elif hasattr(module, 'reset_parameters') and isinstance(module.reset_parameters, Callable):
                    module.reset_parameters()
                else:
                    raise ValueError(
                        f'Object `{obj_name}` does not have a ``param_init_fn`` or a ``reset_parameters`` function. '
                        'This leaves parameters without initialization. Please add a ``param_init_fn`` or ``reset_parameters`` '
                        f'to module `{obj_name}`.',
                    )

            def lambda_fn(module: torch.nn.Module) -> Union[bool, dict]:
                ret = False
                if hasattr(module, '_fsdp_wrap'):
                    ret = bool(module._fsdp_wrap)
                elif hasattr(obj, 'fsdp_wrap_fn') and isinstance(obj.fsdp_wrap_fn, Callable):
                    ret = obj.fsdp_wrap_fn(module)
                    if isinstance(ret, dict):
                        ret = set_custom_fsdp_module_kwargs(ret, process_group_cache)
                return ret

            _auto_wrap_policy = CustomPolicy(lambda_fn)

            fsdp_obj = FullyShardedDataParallel(
                obj,
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                auto_wrap_policy=_auto_wrap_policy,  # type: ignore FSDP type bug
                cpu_offload=cpu_offload,
                mixed_precision=mixed_precision,
                backward_prefetch=backward_prefetch,
                ignored_modules=ignored_modules,
                param_init_fn=_param_init_fn,
                device_id=torch.cuda.current_device(),
                sync_module_states=sync_module_states,
                forward_prefetch=forward_prefetch,
                limit_all_gathers=limit_all_gathers,
                use_orig_params=use_orig_params,
                **kwargs,
            )

            if te_shard_fp8_weight:
                try:
                    from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp
                except ModuleNotFoundError:
                    raise ModuleNotFoundError('Please install transformer-engine to use prepare_te_modules_for_fsdp')
                log.info(f'Calling prepare_te_modules_for_fsdp to enable TE weights sharding')
                prepare_te_modules_for_fsdp(fsdp_obj)

            # The following sync hooks are added to prevent FSDP deadlocks that are caused when some ranks OOM
            # and other ranks do not OOM, leading to OOMing ranks calling all_reduce to wait on the non-OOMing
            # ranks and the non-OOMing ranks calling all_gatherbase to continue with FSDP training:
            #
            #   forward_pre_hook: before forwards of FSDP modules
            #   full_backward_pre_hook: before backwards of FSDP modules
            #   full_backward_hook: before a prefetched unshard called by FSDP's `post_backward_reshard`
            if auto_microbatching:
                for _, module in fsdp_obj.named_modules():
                    if isinstance(module, FullyShardedDataParallel):
                        hook_handles.append(module.register_forward_pre_hook(sync_hook, prepend=True))
                        hook_handles.append(module.register_full_backward_pre_hook(sync_hook, prepend=True))
                    else:
                        hook_handles.append(module.register_full_backward_hook(sync_hook))
                fsdp_obj_named_modules.update(dict(fsdp_obj.named_modules()))

            if hasattr(fsdp_obj, '_exec_order_data'):
                if hasattr(fsdp_obj._exec_order_data, '_forward_prefetch_limit'):
                    fsdp_obj._exec_order_data._forward_prefetch_limit = fsdp_config.forward_prefetch_limit
                else:
                    warnings.warn(
                        'FSDP._exec_order_data does not have attribute _forward_prefetch_limit '
                        'which is unexpected and will result in `forward_prefetch_limit` from FSDP '
                        'config being ignored. Please open an issue to Composer to report this.',
                    )
                if hasattr(fsdp_obj._exec_order_data, '_backward_prefetch_limit'):
                    fsdp_obj._exec_order_data._backward_prefetch_limit = fsdp_config.backward_prefetch_limit
                else:
                    warnings.warn(
                        'FSDP._exec_order_data does not have attribute _backward_prefetch_limit '
                        'which is unexpected and will result in `backward_prefetch_limit` from FSDP '
                        'config being ignored. Please open an issue to Composer to report this.',
                    )
            else:
                warnings.warn(
                    'FSDP does not have attribute _exec_order_data which is unexpected and will '
                    'result in `forward_prefetch_limit` and `backward_prefetch_limit` from FSDP '
                    'config being ignored. Please open an issue to Composer to report this.',
                )

            # Activation Checkpointing
            if activation_checkpointing or activation_cpu_offload:
                # FP8 TE requires using the TE checkpoint function, FSDP activation checkpointing only works with TE non-reentrant checkpointing
                if te_checkpoint_wrapper:
                    assert not activation_checkpointing_reentrant, 'TE checkpoint only works with non-reentrant checkpointing'
                if not activation_checkpointing_reentrant:
                    if te_checkpoint_wrapper:
                        try:
                            import transformer_engine.pytorch as te
                        except ModuleNotFoundError:
                            raise ModuleNotFoundError('Please install transformer-engine to use TE checkpoint wrapper',)

                        # RNG state tracker for checkpointing
                        CUDA_RNG_STATES_TRACKER = te.distributed.CudaRNGStatesTracker()
                        CUDA_RNG_STATES_TRACKER.add('fsdp-rng', te_rng_seed)

                        def get_cuda_rng_tracker():
                            return CUDA_RNG_STATES_TRACKER

                        first_wrap_fn = lambda m: checkpoint_wrapper(
                            m,
                            context_fn=te.distributed.get_activation_recompute_contexts,
                            checkpoint_fn=te.distributed.checkpoint,
                            use_reentrant=False,
                            get_rng_state_tracker=get_cuda_rng_tracker,
                        )
                    else:
                        first_wrap_fn = lambda m: checkpoint_wrapper(
                            m,
                            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                        ) if activation_checkpointing else (lambda module: module)
                    second_wrap_fn = (
                        lambda module: offload_wrapper(
                            first_wrap_fn(module)
                            if activation_checkpointing else module,  # type: ignore reportGeneralTypeIssues
                        )
                    ) if activation_cpu_offload else first_wrap_fn
                else:

                    first_wrap_fn = lambda m: checkpoint_wrapper(
                        m,
                        checkpoint_impl=CheckpointImpl.REENTRANT,
                    ) if activation_checkpointing else (lambda module: module)
                    second_wrap_fn = (
                        lambda module: offload_wrapper(
                            first_wrap_fn(module)
                            if activation_checkpointing else module,  # type: ignore reportGeneralTypeIssues
                        )
                    ) if activation_cpu_offload else first_wrap_fn

                # Choose which modules to activation checkpoint according to the following priority:
                # If module has attribute `module._activation_checkpointing = ...`, always respect it
                # Otherwise checkpoint if root object `obj.activation_checkpointing_fn(module)` is true
                def _check_fn(module: torch.nn.Module) -> bool:
                    if isinstance(module, FullyShardedDataParallel):
                        return False
                    if hasattr(module, '_activation_checkpointing'):
                        return bool(module._activation_checkpointing)
                    if hasattr(
                        obj,
                        'activation_checkpointing_fn',
                    ) and isinstance(obj.activation_checkpointing_fn, Callable):
                        return obj.activation_checkpointing_fn(module)
                    return False

                apply_activation_checkpointing(
                    fsdp_obj,
                    checkpoint_wrapper_fn=second_wrap_fn,  # type: ignore
                    check_fn=_check_fn,  # type: ignore
                )

            setattr(model, obj_name, fsdp_obj)

    # Print FSDP wrapped model and FSDP config if `verbose=True`
    if fsdp_config.verbose:
        log.info(f'FSDP: Wrapped model: {model}')
        log.info(f'FSDP: Using sharding_strategy={sharding_strategy}')
        log.info(f'FSDP: Using cpu_offload={cpu_offload}')
        log.info(f'FSDP: Using mixed_precision={mixed_precision}')
        log.info(f'FSDP: Using backward_prefetch={backward_prefetch}')
        log.info(f'FSDP: Using activation_checkpointing={activation_checkpointing}')
        log.info(f'FSDP: Using activation_cpu_offload={activation_cpu_offload}')
        log.info(f'FSDP: Using te_checkpoint_wrapper={te_checkpoint_wrapper}')
        log.info(f'FSDP: Using te_shard_fp8_weight={te_shard_fp8_weight}')
        log.info(f'FSDP: Using sync_module_states={sync_module_states}')
        log.info(f'FSDP: Using forward_prefetch={forward_prefetch}')
        log.info(f'FSDP: Using limit_all_gathers={limit_all_gathers}')
        log.info(f'FSDP: Using state_dict_type={state_dict_type}')
        log.info(f'FSDP: Using sharded_ckpt_prefix_dir={sharded_ckpt_prefix_dir}')

    # Rebuild optimizer now that parameters are sharded
    if optimizers:
        optim = ensure_tuple(optimizers)[0]
        optim.param_groups.clear()

        if single_param_group_opt_info is not None:
            single_param_group_opt_info.update({'params': list(model.parameters())})
            optim.add_param_group(single_param_group_opt_info)
        elif fsdp_config.use_orig_params:
            assert param_name_to_group_num is not None
            assert group_num_to_opt_group_info is not None

            param_groups = _recreate_fsdp_param_groups_from_unwrapped_opt_info(
                model.named_parameters(),
                param_name_to_group_num,
                group_num_to_opt_group_info,
            )
            for param_group in param_groups:
                optim.add_param_group(param_group)

    return hook_handles, fsdp_obj_named_modules
