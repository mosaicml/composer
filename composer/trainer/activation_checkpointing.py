# Source code copied from:
# - https://github.com/pytorch/pytorch/blob/5aa84c16dbb9640da738866f0d52f1dd0d285f77/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py
# - https://github.com/pytorch/pytorch/blob/0336308be5c2d019b99ed5fb59ec1bf01f735a99/torch/distributed/utils.py
# - https://github.com/pytorch/pytorch/blob/0336308be5c2d019b99ed5fb59ec1bf01f735a99/torch/distributed/fsdp/wrap.py
# - https://github.com/pytorch/pytorch/blob/caa0ab557dd10e04ca413c1508f76ec8ae5adea3/torch/utils/checkpoint.py
# TODO: once torch 1.13 is released, import functions as needed directly from torch and delete this file.

import warnings
import weakref
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, Iterable, Iterator, List, Set, Tuple, cast

import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu

##############################################################################
######### torch.distributed.algorithms._checkpoint/checkpoint_wrapper ########
##############################################################################

_CHECKPOINT_PREFIX = '_checkpoint_wrapped_module'


class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()


class CheckpointWrapper(torch.nn.Module):
    """
    An nn.Module that wraps another nn.Module with checkpointing. Note that this
    module is not meant to be used directly, but instead it is to be used
    through the ``checkpoint_wrapper`` function.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
        offload_to_cpu: bool = False,
        checkpoint_fn=None,
        *checkpoint_fn_args,
        **checkpoint_fn_kwargs,
    ):
        super().__init__()
        self._checkpoint_wrapped_module = mod
        self.checkpoint_impl = checkpoint_impl
        self.offload_to_cpu = offload_to_cpu
        if self.offload_to_cpu:
            self.checkpoint_fn = None
        else:
            if checkpoint_fn is None:
                # use torch.utils.checkpoint
                self.checkpoint_fn = partial(
                    checkpoint,
                    use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                )
            else:
                self.checkpoint_fn = partial(
                    checkpoint_fn,
                    *checkpoint_fn_args,
                    **checkpoint_fn_kwargs,
                )
        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook, with_module=True)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._checkpoint_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def forward(self, *args, **kwargs):
        if self.offload_to_cpu:
            with save_on_cpu(pin_memory=True):
                return self._checkpoint_wrapped_module(*args, **kwargs)
        else:
            # Support keyword arguments for reentrant checkpoint. Note that this
            # only works if user has specified self.checkpoint_impl and is not
            # using their own custom checkpoint_fn.
            if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
                # Pack the args and kwargs
                flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

                # Function that only takes (packed) args, but can unpack them
                # into the original args and kwargs for the checkpointed
                # function, and runs that function.
                def my_function(*inputs):
                    # unpack back into args and kwargs
                    unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                    # run original module
                    return self._checkpoint_wrapped_module(*unpacked_args, **unpacked_kwargs)

                # Pass the function that only takes packed args into reentrant
                # checkpoint API.
                return self.checkpoint_fn(  # type: ignore[misc]
                    my_function,
                    *flat_args,
                )
            else:
                return self.checkpoint_fn(  # type: ignore[misc]
                    self._checkpoint_wrapped_module, *args, **kwargs)

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Overrides :meth:`named_parameters()` to intercept parameter names and
        remove all occurrences of _CHECKPOINT_PREFIX.
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield param_name.replace(f'{_CHECKPOINT_PREFIX}.', ''), param

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> Dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this
        FSDP module is executed. For ``checkpoint_wrapper``, it will strip
        checkpoint-wrapped module prefix so that this module can be loaded into
        non-checkpointed modules. It would still be able to be loaded into
        checkpoint-wrapped modules as this class adds the prefix back before
        loading the state_dict.
        """
        _replace_by_prefix(state_dict, f'{prefix}{_CHECKPOINT_PREFIX}.', prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()``
        is called. For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f'{_CHECKPOINT_PREFIX}.')


def _wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
    assert wrapper_cls is not None
    if hasattr(module, '_wrap_overrides'):
        # If module has a _wrap_overrides attribute, we force overriding the
        # FSDP config with these attributes for this module. Currently this
        # is only used to disable mixed precision for BatchNorm when
        # auto_wrapping.
        overrides = {**kwargs, **module._wrap_overrides}  # type: ignore[arg-type]
        return wrapper_cls(module, **overrides)

    return wrapper_cls(module, **kwargs)


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
    offload_to_cpu: bool = False,
    checkpoint_fn=None,
    *checkpoint_fn_args,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """
    A convenience wrapper for activation checkpointing. If the module is wrapped
    with this function, all subsequent calls to the module will automatically
    perform checkpointing without the user having to explicitly call ``checkpoint``
    function.
    Usage::
        checkpointed_module = checkpoint_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
        checkpoint_impl (Optional[CheckpointImpl]):
            The checkpointing implementation to use. Note that this will only
            be passed into the ``torch.utils.checkpoint.checkpoint``
            implementation, and is ignored if a custom ``checkpoint_fn`` is
            specified. Note that for implementations using reentrant checkpoint
            from ``torch.utils.checkpoint``, keyword arguments will only be
            supported if ``checkpoint_impl`` is passed as ``CheckpointImpl.REENTRANT`.
        offload_to_cpu (Optional[bool]):
            Whether to offload activations of this wrapped module to CPU. Note
            that if this is specified, ``checkpoint_impl`` and ``checkpoint_fn``
            arguments will be ignored in favor of the activations being
            offloaded to CPU. Default is ``False``. Wrappers with activation
            offload can be composed with ones that do recomputation-based
            checkpoint to trade off increased compute versus increased CPU
            memory usage and additional H2D transfers.
        checkpoint_fn (Optional[Callable]):
            Functional checkpoint implementation to use. If this is specified,
            it will be used over the default ``torch.utils.checkpoint.checkpoint``
            implementation and the `checkpoint_impl` argument will be ignored.
        *checkpoint_fn_args: (Sequence[Any]): Arguments to pass into `checkpoint_fn`.
        **checkpoint_fn_kwargs: (Dict[str, Any]): Keyword arguments to pass into `checkpoint_fn`.

    Returns:
        (nn.Module):
            Wrapped module
    """

    return CheckpointWrapper(module, checkpoint_impl, offload_to_cpu, checkpoint_fn, checkpoint_fn_args,
                             checkpoint_fn_kwargs)


def apply_activation_checkpointing_wrapper(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True):
    """
    Applies :func:`checkpoint_wrapper` to modules within `model` based on a user-defined
    configuration. For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :class:`CheckpointWrapper`.
    Usage::
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
        )
        check_fn = lambda l: isinstance(l, nn.Linear)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
    Args:
        module (nn.Module):
            The model who's submodules (or self) should be wrapped with activation checkpointing.
        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
            A `Callable` which will wrap modules
        check_fn (Optional[Callable[nn.Module, nn.Module]])
            A lambda function which will be passed current layer and returns
            ``True`` or ``False`` depending on whether input layer should be wrapped.
    Returns: None (`model` is modified inplace)
    """
    return _recursive_wrap(module=model,
                           auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
                           wrapper_cls=checkpoint_wrapper_fn,
                           ignored_modules=set(),
                           ignored_params=set(),
                           only_wrap_children=True)


##############################################################################
########################## torch.distributed.utils ###########################
##############################################################################


def _pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite)
    Inspiration: https://github.com/facebookresearch/fairscale/blob/eeb6684/fairscale/internal/containers.py#L70
    Usage::
        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    Returns:
        Tuple[Tuple[Any, ...], Tuple[str, ...]]: The first tuple element gives
        gives both positional args and kwarg values, where the positional args
        proceed kwarg values and kwarg values are ordered consistently with the
        kwarg keys. The second tuple element gives the kwarg keys.
        The second tuple element's length is at most the first tuple element's length.
    """
    kwarg_keys: List[str] = []
    flat_args: List[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)


def _unpack_kwargs(flat_args: Tuple[Any, ...], kwarg_keys: Tuple[str, ...]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """See _pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), f'too many keys {len(kwarg_keys)} vs. {len(flat_args)}'
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[:-len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys):])}
    return args, kwargs


def _replace_by_prefix(
    state_dict: Dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).
    Usage::
        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError('old_prefix and new_prefix must be distinct')
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix):]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


##############################################################################
######################## torch.distributed.fsdp.wrap #########################
##############################################################################


def lambda_auto_wrap_policy(module: nn.Module, recurse: bool, unwrapped_params: int, lambda_fn: Callable) -> bool:
    """
    A convenient auto wrap policy to wrap submodules based on an arbitrary user
    function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
    a `wrapper_cls` unit.
    Return if a module should be wrapped during auto wrapping.
    The first three parameters are required by :func:`_recursive_wrap`.
    Args:
       module (nn.Module):
           The module to be considered in this decision.
       recurse (bool):
           Indicate if this is called to make a decision on whether we
           should recurse down a subgraph of the module structure.
           If False, it means this function is called to make a decision
           on whether we should wrap the said module.
       unwrapped_params (int):
           The number of parameters yet to be wrapped in this module.
       lambda_fn (Callable[nn.Module] -> bool):
           If this returns ``True``, this module will be wrapped by
           wrapper_cls individually.
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return lambda_fn(module)


def _recursive_wrap(module: nn.Module,
                    auto_wrap_policy: Callable,
                    wrapper_cls: Callable,
                    ignored_modules: Set[nn.Module],
                    ignored_params: Set[nn.Parameter],
                    only_wrap_children: bool = False,
                    **kwargs: Any) -> Tuple[nn.Module, int]:
    """
    Automatically wrap child modules of *module* that meet the given
    criteria with :func:`auto_wrap`. Does not rely on _ConfigAutoWrap.
    Args:
        module (nn.Module):
            module to recursively wrap
        auto_wrap_policy (Callable):
            A callable specifying a policy to recursively wrap layers with FSDP.
        ignored_modules (Set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
    Returns:
        (nn.Module, int):
            Wrapped module and the number parameters wrapped recursively.
    """
    assert auto_wrap_policy is not None, 'Must specify auto_wrap_policy.'
    assert wrapper_cls is not None, 'Must specify wrapper_cls'
    # Make sure no child is already wrapped.
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            assert not isinstance(child, cast(type, wrapper_cls))
        except TypeError:
            # wrapper_cls is a function as opposed to a class type, just bypass above check.
            pass

    # We count all params, assuming none of them are already wrapped.
    num_params = sum(p.numel() for p in module.parameters() if p not in ignored_params)

    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, unwrapped_params=num_params):
        total_wrapped_params = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _recursive_wrap(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                **kwargs,
            )
            setattr(module, name, wrapped_child)
            # Keep track of how many parameters have been wrapped
            total_wrapped_params += num_wrapped_params
        # decide if we need to wrap the current module,
        # since the left over parameters exceed the number of params to wrap
        remainder = num_params - total_wrapped_params
        if not only_wrap_children and auto_wrap_policy(module=module, recurse=False, unwrapped_params=remainder):
            # Leaf node or final wrapping of the remainder both happen here.
            return _wrap(module, wrapper_cls, **kwargs), num_params
        else:
            return module, total_wrapped_params
    return module, 0


##############################################################################
########################### torch.utils.checkpoint ###########################
##############################################################################


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError('Only tuple of tensors is supported. Got Unsupported input type: ', type(inputs).__name__)


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn('None of the inputs have requires_grad=True. Gradients will be None')


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(set(arg.get_device() for arg in args if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states) -> None:
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)


def _get_autocast_kwargs():
    gpu_autocast_kwargs = {
        'enabled': torch.is_autocast_enabled(),
        'dtype': torch.get_autocast_gpu_dtype(),
        'cache_enabled': torch.is_autocast_cache_enabled()
    }

    cpu_autocast_kwargs = {
        'enabled': torch.is_autocast_cpu_enabled(),
        'dtype': torch.get_autocast_cpu_dtype(),
        'cache_enabled': torch.is_autocast_cache_enabled()
    }

    return gpu_autocast_kwargs, cpu_autocast_kwargs


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('Checkpointing is not compatible with .grad() or when an `inputs` parameter'
                               ' is passed to .backward(). Please use .backward() and do not pass its `inputs`'
                               ' argument.')
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), \
                 torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                 torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError('none of output has requires_grad=True,'
                               ' this checkpoint() is not necessary')
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)

        return (None, None) + grads


def checkpoint(function, *args, use_reentrant: bool = True, **kwargs):
    r"""Checkpoint a model or part of the model
    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.
    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.
    The output of :attr:`function` can contain non-Tensor values and gradient
    recording is only performed for the Tensor values. Note that if the output
    consists of nested structures (ex: custom objects, lists, dicts etc.)
    consisting of Tensors, these Tensors nested in custom structures will not
    be considered as part of autograd.
    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.
    .. warning::
        If ``use_reentrant=True`` is specified, then if the checkpointed segment
        contains tensors detached from the computational graph by `detach()` or
        `torch.no_grad()`, the backward pass will raise an error. This is
        because `checkpoint` makes all the outputs require gradients which
        causes issues when a tensor is defined to have no gradient in the model.
        To circumvent this, detach the tensors outside of the `checkpoint`
        function. Note that the checkpointed segment can contain tensors
        detached from the computational graph if ``use_reentrant=False`` is
        specified.
    .. warning::
        If ``use_reentrant=True`` is specified, at least one of the inputs needs
        to have :code:`requires_grad=True` if grads are needed for model inputs,
        otherwise the checkpointed part of the model won't have gradients. At
        least one of the outputs needs to have :code:`requires_grad=True` as
        well. Note that this does not apply if ``use_reentrant=False`` is
        specified.
    .. warning::
        If ``use_reentrant=True`` is specified, checkpointing currently only
        supports :func:`torch.autograd.backward` and only if its `inputs`
        argument is not passed. :func:`torch.autograd.grad`
        is not supported. If ``use_reentrant=False`` is specified, checkpointing
        will work with :func:`torch.autograd.grad`.
    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool, optional): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad`` and support for
            keyword arguments input into the checkpointed function. Note that future
            versions of PyTorch will default to ``use_reentrant=False``.
            Default: ``True``
        args: tuple containing inputs to the :attr:`function`
    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs and use_reentrant:
        raise ValueError('Unexpected keyword arguments: ' + ','.join(arg for arg in kwargs))

    if use_reentrant:
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        return _checkpoint_without_reentrant(
            function,
            preserve,
            *args,
            **kwargs,
        )


def _checkpoint_without_reentrant(function, preserve_rng_state=True, *args, **kwargs):
    """Checkpointining without re-entrant autograd
    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    gpu_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs()

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_cuda_in_fwd = False
        if torch.cuda._initialized:
            had_cuda_in_fwd = True
            fwd_gpu_devices, fwd_gpu_states = get_device_states(*args)

    # Custom class to be able to take weak references
    class Holder():
        pass

    # The Holder object for each of the saved object is saved directly on the
    # SavedVariable and is cleared when reset_data() is called on it. We MUST make
    # sure that this is the only object having an owning reference to ensure that
    # the Tensor stored in storage is deleted as soon as the corresponding SavedVariable
    # data is cleared.
    storage: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    weak_holder_list = []

    def pack(x):
        # TODO(varal7): Instead of returning abstract object, we can return things metadata (such as
        # size, device, ...) to catch certain cases of undeterministic behavior of the forward
        res = Holder()
        weak_holder_list.append(weakref.ref(res))
        return res

    def unpack(x):
        unpack_counter = 0
        if len(storage) == 0:

            def inner_pack(inner):
                nonlocal unpack_counter
                unpack_counter += 1
                # If the holder went out of scope, the SavedVariable is dead and so
                # the value will never be read from the storage. Skip filling it.
                if weak_holder_list[unpack_counter - 1]() is None:
                    return
                # Use detach here to ensure we don't keep the temporary autograd
                # graph created during the second forward
                storage[weak_holder_list[unpack_counter - 1]()] = inner.detach()
                return

            def inner_unpack(packed):
                raise RuntimeError('You are calling backwards on a tensor that is never exposed. Please open an issue.')

            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if preserve_rng_state and had_cuda_in_fwd:
                rng_devices = fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state):
                if preserve_rng_state:
                    torch.set_rng_state(fwd_cpu_state)
                    if had_cuda_in_fwd:
                        set_device_states(fwd_gpu_devices, fwd_gpu_states)

                with torch.enable_grad(), \
                     torch.cuda.amp.autocast(**gpu_autocast_kwargs), \
                     torch.cpu.amp.autocast(**cpu_autocast_kwargs), \
                     torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args, **kwargs)

        if x not in storage:
            raise RuntimeError('Attempt to retrieve a tensor saved by autograd multiple times without checkpoint'
                               ' recomputation being triggered in between, this is not currently supported. Please'
                               ' open an issue with details on your use case so that we can prioritize adding this.')

        return storage[x]

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args, **kwargs)
        if torch.cuda._initialized and preserve_rng_state and not had_cuda_in_fwd:  # type: ignore
            # Cuda was not initialized before running the forward, so we didn't
            # stash the CUDA state.
            raise RuntimeError("PyTorch's CUDA state was initialized in the forward pass "
                               'of a Checkpoint, which is not allowed. Please open an issue '
                               'if you need this feature.')

    return output
