# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running distributed data parallel training."""

import collections
import logging
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast

import torch
from packaging import version
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric, MetricCollection

from composer.core import Precision, State
from composer.devices import Device
from composer.trainer.meta_safe_apply import meta_safe_apply
from composer.trainer.mosaic_fsdp import patch_pytorch
from composer.trainer.mosaic_fsdp_utils import BACKWARD_PREFETCH_MAP, SHARDING_MAP, get_cpu_offload, get_mixed_precision
from composer.utils import StringEnum, dist, ensure_tuple, using_torch_2

__all__ = ['DDPSyncStrategy', 'ddp_sync_context', 'prepare_ddp_module', 'prepare_fsdp_module']

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

    raise RuntimeError('When the world size is > 1, ``torch.distributed`` must be used. However, it is '
                       'not available in your installation of PyTorch. Please install or build PyTorch '
                       'with distributed support.')


def set_fsdp_default(fsdp_config: Dict[str, Any]):
    """Modify fsdp_config to set default values for missing keys."""
    fsdp_config.setdefault('activation_checkpointing', False)
    fsdp_config.setdefault('activation_checkpointing_reentrant', True)
    fsdp_config.setdefault('activation_cpu_offload', False)
    fsdp_config.setdefault('backward_prefetch', 'BACKWARD_POST')
    fsdp_config.setdefault('cpu_offload', False)
    fsdp_config.setdefault('flatten_parameters', True)
    fsdp_config.setdefault('forward_prefetch', False)
    fsdp_config.setdefault('ignored_modules', None)
    fsdp_config.setdefault('keep_low_precision_grads', False)
    fsdp_config.setdefault('limit_all_gathers', True)
    fsdp_config.setdefault('load_monolith_rank0_only', False)
    fsdp_config.setdefault('load_planner', None)
    fsdp_config.setdefault('mixed_precision', 'DEFAULT')
    fsdp_config.setdefault('save_planner', None)
    fsdp_config.setdefault('sharded_ckpt_prefix_dir', 'ep{epoch}-ba{batch}')
    fsdp_config.setdefault('sharding_strategy', 'FULL_SHARD')
    fsdp_config.setdefault('state_dict_type', 'full')
    fsdp_config.setdefault('sync_module_states', False)
    fsdp_config.setdefault('use_orig_params', True)
    fsdp_config.setdefault('verbose', False)
    return fsdp_config


def _recreate_fsdp_param_groups_from_unwrapped_opt_info(
        fsdp_wrapped_named_params: Iterator[Tuple[str,
                                                  torch.nn.Parameter]], non_wrapped_param_names_to_group_num: Dict[str,
                                                                                                                   int],
        group_num_to_optimizer_info: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        non_wrapped_param_names_to_group_num: a Dict mapping from the original model param names
                                            to the param group number
        group_num_to_optimizer_info: stores info like lr, eps for each group

    Returns a list of param groups, referencing the fsdp parameters
    """
    is_torch_2_0 = using_torch_2()
    if not is_torch_2_0:
        raise RuntimeError('Helper function is only supported in torch 2.0')

    from torch.distributed.fsdp._common_utils import clean_tensor_name

    # initialize an empty list of parameters for each optimizer group
    for group_num in group_num_to_optimizer_info.keys():
        group_num_to_optimizer_info[group_num]['params'] = []

    for fsdp_name, param in fsdp_wrapped_named_params:

        unwrapped_name = clean_tensor_name(fsdp_name)
        # need to have a 1:1 mapping between a fsdp param name and the non-wrapped vanilla param name
        retrieved_group_num = non_wrapped_param_names_to_group_num[unwrapped_name]
        group_num_to_optimizer_info[retrieved_group_num]['params'].append(param)

    # return sorted optimizer info groups
    return [group_num_to_optimizer_info[num] for num in sorted(group_num_to_optimizer_info.keys())]


def prepare_fsdp_module(
    model: torch.nn.Module,
    optimizers: Optional[Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]],
    fsdp_config: Dict[str, Any],
    precision: Precision,
    device: Device,
    auto_microbatching: bool,
) -> None:
    """Prepare a module (assumed ComposerModel) and optimizer for use with :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    Args:
        model (torch.nn.Module): The model to wrap.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): The optimizer for `model`, assumed to have a single param group := model.parameters().
        fsdp_config (Dict[str, Any]): The FSDP config.
        precision: (Precision): The precision being used by the Trainer, used to fill in defaults for FSDP `mixed_precision` settings.
        device (Device): The device being used by the Trainer.
        auto_microbatching (bool, optional): Whether or not auto microbatching is enabled.
    """
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        raise RuntimeError('To use FSDP with Composer, you must use torch>=1.13.0.')
    is_torch_2_0 = using_torch_2()
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl,
                                                                             apply_activation_checkpointing,
                                                                             checkpoint_wrapper)
    from torch.distributed.fsdp import FullyShardedDataParallel
    if not is_torch_2_0:
        from torch.distributed.fsdp.flatten_params_wrapper import FlattenParamsWrapper

    patch_pytorch()

    set_fsdp_default(fsdp_config)

    # Check sync_module_states is True for mixed initialization
    if fsdp_config['sync_module_states'] == False:
        rank_on_meta = 1 if next(model.parameters()).device.type == 'meta' else 0
        all_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
        dist.all_reduce(all_ranks_meta, reduce_operation='MIN')
        any_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
        dist.all_reduce(any_ranks_meta, reduce_operation='MAX')
        if all_ranks_meta.item() == 0 and any_ranks_meta.item() == 1:
            raise ValueError('Detected mixed initialization where some ranks have model on cpu or '
                             'gpu and some ranks are on meta. Either keep all ranks on the same '
                             "device or set fsdp_config['sync_module_states'] = True. Otherwise, "
                             'some weights may be randomly initialized when loading a checkpoint.')

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

    kwargs = {}
    if is_torch_2_0:
        # Support of new parameter `use_orig_params` in PyTorch 2.0 or higher.
        # Setting this to `True` has FSDP use `module`'s original parameters via method
        # `nn.Module.named_parameters` instead of FSDP's internal class `FlatParameter`. However,
        # setting it to `False` exposes FSDP's internal class `FlatParameter` via method
        # `nn.Module.named_parameters`.
        # Setting it to `True` is mandatory when using `torch.compile()`.
        kwargs['use_orig_params'] = fsdp_config['use_orig_params']

    # necessary variables for optimizers with multiple param groups in FSDP
    num_param_groups = None
    param_name_to_group_num = None
    group_num_to_param_group_info = None

    optimizer_specific_info = None
    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        if len(optimizers_tuple) != 1:
            raise NotImplementedError(f'Only one optimizer is supported; found {len(optimizers_tuple)} optimizers')

        # clearing optimizer param groups and state
        # that will be recreated at the end of prepare_fsdp_module
        optim = optimizers_tuple[0]

        num_param_groups = len(optim.param_groups)
        if num_param_groups > 1:
            if not (is_torch_2_0 and kwargs['use_orig_params']):
                raise RuntimeError('Multiple optimizer groups with FSDP are only supported on torch 2.0 \
                                   with use_orig_params=True.')
            # optimizer.param_groups do not contain parameter names which are needed
            # to keep track of the different parameters in each group
            # so we use the pointers between model.parameters() and model.named_parameters()
            # to get the names of the parameters within optimizer.param_groups
            param_pointer_to_param_name = {id(p): n for n, p in model.named_parameters()}
            param_name_to_group_num = {}
            group_num_to_param_group_info = {}
            for group_num in range(len(optim.param_groups)):
                # Need to in-line to avoid a reference which causes FSDP to allocate extra GPU memory
                # group = optim.param_groups[group_num]
                for param_num in range(len(optim.param_groups[group_num]['params'])):
                    # Need to in-line to avoid a reference which causes FSDP to allocate extra GPU memory
                    # param = optim.param_groups[group_num]['params'][param_num]
                    param_name_to_group_num[param_pointer_to_param_name[id(
                        optim.param_groups[group_num]['params'][param_num])]] = group_num

                # this includes optimizer-specific values like lr, eps
                # this will be used as the kwargs for the optim param groups later
                optimizer_specific_group_info = {
                    k: v for k, v in optim.param_groups[group_num].items() if k != 'params'
                }
                group_num_to_param_group_info[group_num] = optimizer_specific_group_info
        else:
            optimizer_specific_info = {k: v for k, v in optim.param_groups[0].items() if k != 'params'}

        optim.param_groups.clear()
        optim.state.clear()

    sharding_map_key = fsdp_config['sharding_strategy'].upper()
    sharding_strategy = SHARDING_MAP[sharding_map_key]

    cpu_offload = get_cpu_offload(cpu_offload=fsdp_config['cpu_offload'])

    mixed_precision = fsdp_config['mixed_precision']
    keep_low_precision_grads = fsdp_config['keep_low_precision_grads']
    mixed_precision, param_dtype, _, _ = get_mixed_precision(precision,
                                                             mixed_precision=mixed_precision,
                                                             keep_low_precision_grads=keep_low_precision_grads)

    # Note: FSDP does support the use of torch.float32 with sharding.
    # They just never expected a user to pass in torch.float32 into mixed_precision as a param_dtype.
    # See: https://github.com/pytorch/pytorch/issues/90584
    # The PR fixing this bug is merged into PyTorch, but it hasn't made its way into a release yet.
    # Instead a user needs to pass in `None` as param_dtype to have the parameters as torch.float32.
    # TODO: remove these checks when PyTorch has a release that includes the fix.
    if sharding_map_key != 'NO_SHARD':
        if (precision == Precision.AMP_FP16 and param_dtype not in [torch.float16, None] or
                precision == Precision.AMP_BF16 and param_dtype not in [torch.bfloat16, None]):
            raise ValueError(
                f'FSDP in PyTorch 1.13 does not support precision `{precision}` with sharding strategy `{sharding_strategy}` '
                f'and param_dtype `{param_dtype}.` Consider using one of the predefined mixed_precision strategies '
                "(choose: `'FULL'`, `'DEFAULT'`, `'PURE'`)")

        if param_dtype == torch.float32:
            raise ValueError(
                f'FSDP in PyTorch 1.13 does not support param_dtype `{param_dtype}` with sharding_strategy `{sharding_map_key}` '
                f'Consider using `amp` or `bf16` for precision or setting param_dtype in mixed_precision to `None` '
                f'with sharding strategy `{sharding_map_key}.`')

    backward_prefetch = BACKWARD_PREFETCH_MAP[fsdp_config['backward_prefetch'].upper()]
    activation_checkpointing = fsdp_config['activation_checkpointing']
    activation_cpu_offload = fsdp_config['activation_cpu_offload']
    sync_module_states = fsdp_config['sync_module_states']
    forward_prefetch = fsdp_config['forward_prefetch']
    limit_all_gathers = fsdp_config['limit_all_gathers']
    ignored_modules = fsdp_config['ignored_modules']
    state_dict_type = fsdp_config['state_dict_type']
    activation_checkpointing_reentrant = fsdp_config['activation_checkpointing_reentrant']
    sharded_ckpt_prefix_dir = fsdp_config['sharded_ckpt_prefix_dir']

    # We choose to not wrap the ComposerModel directly, but instead wrap any submodules like `ComposerModel.model`
    # This makes it safer to call ComposerModel-specific functions like 'eval_forward' that
    # may make calls to sharded submodules. If we only wrap the submodules, then any call that ComposerModel makes
    # to a FSDP-wrapped submodule's `forward()` function will be safe and all-gather the necessary weights before `forward()`.
    for obj_name, obj in model.named_children():
        if not isinstance(obj, (Metric, MetricCollection)):

            # Skip wrapping submodules which are explicitly marked with no wrap
            if hasattr(obj, '_fsdp_wrap') and not bool(obj._fsdp_wrap):
                continue

            def _param_init_fn(module: torch.nn.Module) -> None:
                # A dictionary of all tied parameter pointers to module names
                tied_pointers = {}

                # Goes through all modules finding which weights have the same pointers
                for name, mod in module.named_modules():
                    # Since FSDP recursively wraps, at parent modules we can encounter already
                    # wrapped weights, as a result we should skip any modules with `_fsdp_wrapped_module.`
                    if '_fsdp_wrapped_module' in name:
                        continue
                    for attr in ['weight', 'bias']:
                        if hasattr(mod, attr):
                            mod_attr = getattr(mod, attr)
                            if mod_attr is None:
                                continue
                            ptr = id(mod_attr)
                            ptr_attr = (ptr, attr)
                            name_list = tied_pointers.get(ptr_attr, [])
                            name_list.append(name)
                            tied_pointers[ptr_attr] = name_list

                # Creates a dictionary of module names that should be tied together
                tied_mod_names = collections.defaultdict(list)
                # Creates a set of modules we should not initialize
                should_not_init_params = set()
                for ptr_attr_type, mod_names in tied_pointers.items():
                    # No modules for this pointer are tied
                    if len(mod_names) == 1:
                        continue
                    _, attr_type = ptr_attr_type
                    first = next(mod_names.__iter__())
                    for elem in mod_names:
                        should_not_init_params.add('.'.join([elem, attr_type]))
                        tied_mod_names[(first, attr_type)].append(elem)
                    # Make sure at least one of the tied parameters is initialized
                    should_not_init_params.remove('.'.join([first, attr_type]))

                meta_safe_apply(module,
                                lambda t: torch.empty_like(t, device=f'cuda:{torch.cuda.current_device()}'),
                                should_not_init_params,
                                module_name='')

                if len(tied_mod_names) > 0:
                    warnings.warn(('The passed in model appears to have tied weights. In order to '
                                   'support effective weight tying, the tied modules need to be '
                                   'in the same FSDP module. If the weights are not properly tied '
                                   'it can lead to loss spikes. We have tried our best to ensure '
                                   'the tied weights are in the same FSDP module.'))

                # Redoes weight tying
                for name_attr, tied_names in tied_mod_names.items():
                    name, attr = name_attr
                    src_mod = module.get_submodule(name)
                    # We need to make sure the source and destination
                    # modules end up in the same FSDP module otherwise
                    # with sharding weight tying gets violated
                    src_mod._fsdp_wrap = False  # type: ignore
                    src_params = getattr(src_mod, attr)
                    for tied_name in tied_names:
                        dest_mod = module.get_submodule(tied_name)
                        dest_mod._fsdp_wrap = False  # type: ignore
                        setattr(dest_mod, attr, src_params)

                if hasattr(obj, 'param_init_fn') and isinstance(obj.param_init_fn, Callable):
                    module.apply(obj.param_init_fn)
                elif hasattr(module, 'reset_parameters') and isinstance(module.reset_parameters, Callable):
                    module.reset_parameters()
                else:
                    raise ValueError(
                        f'Object `{obj_name}` does not have a ``param_init_fn`` or a ``reset_parameters`` function. '
                        'This leaves parameters without initialization. Please add a ``param_init_fn`` or ``reset_parameters`` '
                        f'to module `{obj_name}`.')

            if version.parse(torch.__version__) > version.parse('2.1.0.dev'):
                # CustomPolicy is only supported in torch v2.1.0-rc1 or higher
                from torch.distributed.fsdp.wrap import CustomPolicy  # type: ignore

                def lambda_fn(module: torch.nn.Module) -> Union[bool, dict]:
                    ret = False
                    if hasattr(module, '_fsdp_wrap'):
                        ret = bool(module._fsdp_wrap)
                    elif hasattr(obj, 'fsdp_wrap_fn') and isinstance(obj.fsdp_wrap_fn, Callable):
                        ret = obj.fsdp_wrap_fn(module)
                        from composer.trainer.mosaic_fsdp_utils import _set_custom_fsdp_module_kwargs
                        if isinstance(ret, dict):
                            ret = _set_custom_fsdp_module_kwargs(ret, process_group_cache)
                    if ret and auto_microbatching:
                        module.register_forward_hook(sync_hook)
                        module.register_full_backward_hook(sync_hook)
                    return ret

                _auto_wrap_policy = CustomPolicy(lambda_fn)
            else:
                # Choose which modules to FSDP wrap according to the following priority:
                # If module has attribute `module._fsdp_wrap = ...`, always respect it
                # Otherwise wrap if root object `obj.fsdp_wrap_fn(module)` is true.
                def __auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
                    if recurse:
                        return True
                    should_be_wrapped = False
                    if hasattr(module, '_fsdp_wrap'):
                        should_be_wrapped = bool(module._fsdp_wrap)
                    elif hasattr(obj, 'fsdp_wrap_fn') and isinstance(obj.fsdp_wrap_fn, Callable):
                        should_be_wrapped = obj.fsdp_wrap_fn(module)

                    if should_be_wrapped and auto_microbatching:
                        module.register_forward_hook(sync_hook)
                        module.register_full_backward_hook(sync_hook)
                    return should_be_wrapped

                if is_torch_2_0:

                    def _auto_wrap_policy_new(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
                        return __auto_wrap_policy(module, recurse, nonwrapped_numel)

                    _auto_wrap_policy = _auto_wrap_policy_new

                else:

                    def _auto_wrap_policy_old(module: torch.nn.Module, recurse: bool, unwrapped_params: int) -> bool:
                        return __auto_wrap_policy(module, recurse, unwrapped_params)

                    _auto_wrap_policy = _auto_wrap_policy_old

            fsdp_obj = FullyShardedDataParallel(
                obj,
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
                **kwargs,
            )

            # Activation Checkpointing
            if activation_checkpointing or activation_cpu_offload:
                if not activation_checkpointing_reentrant:
                    first_wrap_fn = lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                                                                ) if activation_checkpointing else (lambda module:
                                                                                                    module)
                    second_wrap_fn = (
                        lambda module: checkpoint_wrapper(
                            first_wrap_fn(module),  # type: ignore reportGeneralTypeIssues
                            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                            offload_to_cpu=True)) if activation_cpu_offload else first_wrap_fn
                else:
                    first_wrap_fn = checkpoint_wrapper if activation_checkpointing else (lambda module: module)
                    second_wrap_fn = (
                        lambda module: checkpoint_wrapper(
                            first_wrap_fn(module),  # type: ignore reportGeneralTypeIssues
                            offload_to_cpu=True)) if activation_cpu_offload else first_wrap_fn

                # Choose which modules to activation checkpoint according to the following priority:
                # If module has attribute `module._activation_checkpointing = ...`, always respect it
                # Otherwise checkpoint if root object `obj.activation_checkpointing_fn(module)` is true
                def _check_fn(module: torch.nn.Module) -> bool:
                    if not is_torch_2_0 and isinstance(module, FlattenParamsWrapper):
                        return False
                    if isinstance(module, FullyShardedDataParallel):
                        return False
                    if hasattr(module, '_activation_checkpointing'):
                        return bool(module._activation_checkpointing)
                    if hasattr(obj, 'activation_checkpointing_fn') and isinstance(obj.activation_checkpointing_fn,
                                                                                  Callable):
                        return obj.activation_checkpointing_fn(module)
                    return False

                apply_activation_checkpointing(
                    fsdp_obj,
                    checkpoint_wrapper_fn=second_wrap_fn,  # type: ignore
                    check_fn=_check_fn,  # type: ignore
                )

            setattr(model, obj_name, fsdp_obj)

    # Print FSDP wrapped model and FSDP config if `verbose=True`
    if fsdp_config['verbose']:
        print(f'FSDP: Wrapped Model:')
        print(model)
        print(f'FSDP: Using sharding_strategy={sharding_strategy}')
        print(f'FSDP: Using cpu_offload={cpu_offload}')
        print(f'FSDP: Using mixed_precision={mixed_precision}')
        print(f'FSDP: Using backward_prefetch={backward_prefetch}')
        print(f'FSDP: Using activation_checkpointing={activation_checkpointing}')
        print(f'FSDP: Using activation_cpu_offload={activation_cpu_offload}')
        print(f'FSDP: Using sync_module_states={sync_module_states}')
        print(f'FSDP: Using forward_prefetch={forward_prefetch}')
        print(f'FSDP: Using limit_all_gathers={limit_all_gathers}')
        print(f'FSDP: Using state_dict_type={state_dict_type}')
        print(f'FSDP: Using sharded_ckpt_prefix_dir={sharded_ckpt_prefix_dir}')

    # Rebuild optimizer now that parameters are sharded
    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        optim = optimizers_tuple[0]
        optim.param_groups.clear()

        assert num_param_groups is not None
        if num_param_groups > 1:
            assert param_name_to_group_num is not None
            assert group_num_to_param_group_info is not None

            param_groups = _recreate_fsdp_param_groups_from_unwrapped_opt_info(model.named_parameters(),
                                                                               param_name_to_group_num,
                                                                               group_num_to_param_group_info)
            for param_group in param_groups:
                optim.add_param_group(param_group)
        else:
            assert optimizer_specific_info is not None
            optimizer_specific_info.update({'params': list(model.parameters())})
            optim.add_param_group(optimizer_specific_info)
