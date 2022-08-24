from contextlib import suppress
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, Iterator, Set, Tuple, cast

import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _replace_by_prefix
from torch.utils.checkpoint import checkpoint

_CHECKPOINT_PREFIX = "_checkpoint_wrapped_module"


class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()


class CheckpointWrapper(torch.nn.Module):
    """
    An nn.Module that wraps another nn.Module with checkpointing.
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
        offload_mgr = save_on_cpu(pin_memory=True) if self.offload_to_cpu else suppress()
        with offload_mgr:  # type: ignore[attr-defined]
            return self.checkpoint_fn(self._checkpoint_wrapped_module, *args, **kwargs)

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]:
        """
        Overrides :meth:`named_parameters()` to intercept parameter names and
        remove all occurrences of _CHECKPOINT_PREFIX.
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield param_name.replace(f"{_CHECKPOINT_PREFIX}.", ""), param

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
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}.", prefix)
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
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}.")


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
            The checkpointing implementation to use. Currently only
            CheckpointImpl.REENTRANT is supported. Note that this will only
            be passed into the ``torch.utils.checkpoint.checkpoint``
            implementation, and is ignored if a custom ``checkpoint_fn`` is
            specified.
        offload_to_cpu (Optional[bool]):
            Whether to offload outer activations to CPU. Note that this
            currently only works with CheckpointImpl.REENTRANT.
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
    assert auto_wrap_policy is not None, "Must specify auto_wrap_policy."
    assert wrapper_cls is not None, "Must specify wrapper_cls"
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
    # TODO: Importing inside function to avoid circular import issue between FSDP and
    # checkpoint_wrapper. This can be resolved once wrap() APIs are decoupled from FSDP code.
    return _recursive_wrap(module=model,
                           auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
                           wrapper_cls=checkpoint_wrapper_fn,
                           ignored_modules=set(),
                           ignored_params=set(),
                           only_wrap_children=True)
