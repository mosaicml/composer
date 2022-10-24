# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Modify model architectures.

Algorithms, such as :class:`~composer.algorithms.blurpool.BlurPool`, replace model parameters in-place.
This module contains helper functions to replace parameters in :class:`~torch.nn.Module` and
:class:`~torch.optim.Optimizer` instances.

Attributes:
    ReplacementFunction ((torch.nn.Module, int) -> Optional[torch.nn.Module]): Surgery replacement function protocol.

        The function is provided with a :class:`torch.nn.Module` and a counter for the number of
        instances of the module type have been seen. The function should return a replacement
        :class:`torch.nn.Module` if the module type should be replaced, or ``None`` otherwise.

        Args:
            module (torch.nn.Module): Source module
            module_index (int): The i-th instance of module class.

        Returns: Optional[torch.nn.Module]: The replacement module, or ``None`` to indicate no modification.
"""
import collections
import itertools
import logging
import textwrap
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, OrderedDict, Sequence, Tuple, Type, Union

import torch
import torch.distributed
from torch.optim import Optimizer

from composer.utils.iter_helpers import ensure_tuple

log = logging.getLogger(__name__)

__all__ = [
    'ReplacementFunction',
    'replace_module_classes',
    'count_module_instances',
    'update_params_in_optimizer',
]

ReplacementFunction = Callable[[torch.nn.Module, int], Optional[torch.nn.Module]]


def _add_children_recursive(
    module: torch.nn.Module,
    children_to_parents_and_names: OrderedDict[torch.nn.Module, List[Tuple[torch.nn.Module, str]]],
) -> None:
    # recursively build up children_to_parents_and_names so it maps a module to the list of
    # (parent_module, attribute name)
    for name, child in module.named_children():
        if child not in children_to_parents_and_names:
            children_to_parents_and_names[child] = []
            _add_children_recursive(child, children_to_parents_and_names)
        children_to_parents_and_names[child].append((module, name))


# adapted from https://github.com/microsoft/DeepSpeed/blob/b8ff4825aae4bced15a29a4298cb3e59098df999/deepspeed/module_inject/replace_module.py#L699
def replace_module_classes(
    module: torch.nn.Module,
    policies: Mapping[Type[torch.nn.Module], ReplacementFunction],
    optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,
    recurse_on_replacements: bool = False,
    indices: Optional[Dict[Any, int]] = None,
) -> Dict[torch.nn.Module, torch.nn.Module]:
    """Modify model in-place by recursively applying replacement policies.

    .. rubric:: Example

    The following example replaces all convolution layers with linear layers, and linear layers will be replaced if
    there are 16 input features. Recursion occurs on replacement.

    * The first replacement policy replaces the ``nn.Conv2d(1, 32, 3, 1)`` layer with a ``nn.Linear(16, 32)`` layer.
    * The second replacement policy recurses on this replaced layer. Because ``in_features == 16``, this policy
      replaces the layer with a ``nn.Linear(32, 64)``.
    * This policy is invoked again on this new layer. However, since ``in_features == 32``,
      no replacement occurs and this policy returns ``None``.
    * Since all policies do not match or now return ``None`` on all layers, surgery is finished.
    * All replacements, including intermediate replacements, are returned.

    .. testsetup::

        from composer.utils.module_surgery import replace_module_classes

    .. doctest::

        >>> from torch import nn
        >>> module = nn.Sequential(
        ...     nn.Conv2d(1, 32, 3, 1),
        ...     nn.ReLU(),
        ...     nn.MaxPool2d(2),
        ...     nn.Flatten(),
        ...     nn.Linear(5408, 128),
        ...     nn.ReLU(),
        ...     nn.LogSoftmax(dim=1),
        ... )
        >>> policies = {
        ...     nn.Conv2d: lambda x, idx: nn.Linear(16, 32),
        ...     nn.Linear: lambda x, idx: nn.Linear(32, 64) if x.in_features == 16 else None
        ... }
        >>> replace_module_classes(module, policies, recurse_on_replacements=True)
        {Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)): Linear(in_features=16, out_features=32, bias=True), Linear(in_features=16, out_features=32, bias=True): Linear(in_features=32, out_features=64, bias=True)}

    .. warning::

        When a module is replaced, any tensor values within the module are not copied over
        to the new module even when the shape is identical. For example, if model weights
        are initialized prior to calling this function, the initialized weights will not
        be preserved in any replacements.


    Arguments:
        module (torch.nn.Module): Model to modify.
        policies (Mapping[torch.nn.Module, ReplacementFunction]): Mapping of source module class to
            a replacement function. Matching policies are applied in the iteration order of the dictionary, so
            if order is important, an :class:`OrderedDict` should be used. The replacement function may
            return either another :class:`~torch.nn.Module` or ``None``. If the latter, the source module
            is not replaced.
        recurse_on_replacements (bool): If true, policies will be applied to any module returned
            by another policy. For example, if one policy replaces a :class:`~torch.nn.Conv2d`
            with a module containing another :class:`~torch.nn.Conv2d`, the replacement function will
            be invoked with this new child :class:`~torch.nn.Conv2d` instance. If the replacement policies
            are not conditioned on module properties that change during replacement, infinite recursion is
            possible.
        indices (Dict[Any, int], optional): A dictionary mapping module types to the number of times
            they've occurred so far in the recursive traversal of
            ``module`` and its child modules. The value is provided to replacement functions, so they
            may switch behaviors depending on the number of replacements that occurred for a given module type.

            .. note::

                These indices may not correspond to the order in which modules get called in the forward pass.

        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): One or more
            :class:`~torch.optim.Optimizer` objects. If provided,
            this function will attempt to remove parameters in replaced modules
            from these optimizers, and add parameters from the newly-created
            modules. See :func:`update_params_in_optimizer` for more information.

    Returns:
        Dict[torch.nn.Module, torch.nn.Module]:
            A dictionary of ``{original_module: replacement_module}``
            reflecting the replacements applied to ``module`` and its children.
    """
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        raise TypeError(
            textwrap.dedent("""\
                Surgery is not supported after a module is wrapped with
                `torch.nn.parallel.DistributedDataParallel` Instead, please preform surgery on the underlying
                `module.module` and re-wrap the `module.module` with `torch.nn.parallel.DistributedDataParallel`"""))
    try:
        import deepspeed
    except ImportError:
        pass
    else:
        if isinstance(module, deepspeed.DeepSpeedEngine):
            raise TypeError(
                textwrap.dedent("""\
                    Surgery is not supported after a module is wrapped with
                    `deepspeed.DeepSpeedEngine` Instead, please perform surgery on the underlying module`,
                    and re-wrap it with `deepspeed.DeepSpeedEngine`"""))
    replaced_pairs = {}
    children_to_parents_and_names: OrderedDict[torch.nn.Module, List[Tuple[torch.nn.Module,
                                                                           str]]] = collections.OrderedDict()
    _add_children_recursive(module, children_to_parents_and_names)
    indices = indices if indices is not None else {c: 0 for c in policies}

    default_device = _infer_device(module)

    while len(children_to_parents_and_names) > 0:
        child, parents = children_to_parents_and_names.popitem(last=False)
        for policy_class, replacement_fn in policies.items():
            if not isinstance(child, policy_class):
                continue
            module_index = indices[policy_class]
            replacement = replacement_fn(
                child,
                module_index,
            )
            indices[policy_class] += 1
            if replacement is not None:
                assert child not in replaced_pairs

                # if no device inferred (child has no parameters, e.g. Pool2d),
                # use the default device inferred from the entire module.
                device = _infer_device(child)
                if device is None:
                    device = default_device

                if device:
                    replacement = replacement.to(device)

                replaced_pairs[child] = replacement
                for parent, name in parents:
                    # update each parent with the replaced child
                    setattr(parent, name, replacement)

                # recurse on new child object
                if recurse_on_replacements:
                    children_to_parents_and_names[replacement] = list(parents)  # copy the parents list
                    _add_children_recursive(replacement, children_to_parents_and_names)
    if optimizers:
        for old_module, new_module in replaced_pairs.items():
            update_params_in_optimizer(old_params=old_module.parameters(),
                                       new_params=new_module.parameters(),
                                       optimizers=optimizers)
    elif len(replaced_pairs) > 0:
        log.info(
            textwrap.dedent("""\
            optimizers was not provided. Be sure to either create the optimizer after
            invoking this method, or manually add new parameters to the existing optimizer."""))

    return replaced_pairs


def _infer_device(module: torch.nn.Module) -> Optional[torch.device]:
    """Attempt to infer a module's device by inspecting its parameters and buffers."""
    try:
        p = next(itertools.chain(module.parameters(), module.buffers()))
    except StopIteration:
        return None
    else:
        return p.device


def count_module_instances(module: torch.nn.Module, module_class: Union[Type[torch.nn.Module],
                                                                        Tuple[Type[torch.nn.Module], ...]]) -> int:
    """Counts the number of instances of ``module_class`` in ``module``, recursively.

    .. rubric:: Example

    .. testsetup::

        from composer.utils.module_surgery import count_module_instances

    .. doctest::

        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 64), nn.ReLU())
        >>> count_module_instances(module, nn.Linear)
        2
        >>> count_module_instances(module, (nn.Linear, nn.ReLU))
        3

    Args:
        module (torch.nn.Module): The source module.
        module_class (Type[torch.nn.Module] | Tuple[Type[torch.nn.Module], ...]):
            The module type (or tuple of module types) to count.

    Returns:
        int: The number of instances of ``module_class`` in ``module``
    """
    found_instances = set()
    _recur_count_module_instances(module, module_class, found_instances)
    return len(found_instances)


def _recur_count_module_instances(
    module: torch.nn.Module,
    module_class: Union[Type[torch.nn.Module], Tuple[Type[torch.nn.Module], ...]],
    found_instances: set,
):
    """Counts instances of ``module_class`` in ``module``, recursively, using a set to deduplicate.

    We require creating a set of all found modules of instance module_class since a model might
    have duplicate references to a particular module.
    """
    for _, child in module.named_children():
        if isinstance(child, module_class):
            found_instances.add(child)
        _recur_count_module_instances(child, module_class, found_instances)


def _tensor_in(tensor: torch.Tensor, iterable: Iterable[torch.Tensor]):
    """Returns whether ``tensor is element`` for any element in ``iterable``.

    This function is necessary because ``tensor in iterable`` does not work reliably for :class:`.Tensor` objects.

    See https://discuss.pytorch.org/t/how-to-judge-a-tensor-is-in-a-list/15998/4
    for further discussion.
    """
    return any(tensor is elem for elem in iterable)


def _find_param_in_optimizer(param: torch.nn.parameter.Parameter, optimizer: Optimizer) -> int:
    """Returns the index of the optimizer ``param_group`` containing ``param``.

    Optimizers store their parameters within an iterable of ``dict``s called
    :attr:`~torch.optim.Optimizer.param_groups`.
    By default, there is only one group in :attr:`~torch.optim.Optimizer.param_groups`
    that containing all the parameters, but there can be more than one. This
    function is a simple utility to identify which parameter group in
    :attr:`~torch.optim.Optimizer.param_groups` contains a given parameter, if any. The information
    might be desirable to, e.g., inspect the optimizer settings being used
    for a given parameter, or to remove unused parameter tensors from
    the optimizer.

    Args:
        param (torch.nn.parameter.Parameter): The parameter to search for.
        optimizer (torch.optim.Optimizer): The optimizer to search within.

    Returns:
        int: The index within `opt.param_groups` of the first group containing ``param``,
        or `-1` if ``param`` is not in the ``opt`.
    """
    for i, group in enumerate(optimizer.param_groups):
        param_list: List[torch.nn.parameter.Parameter] = group['params']
        if _tensor_in(param, param_list):
            return i

    return -1


def _ordered_diff(first: List, second: List) -> List:
    """Returns first - second while maintaining the order in first."""
    second_list = set(second)
    return [item for item in first if item not in second_list]


def update_params_in_optimizer(old_params: Iterable[torch.nn.parameter.Parameter],
                               new_params: Iterable[torch.nn.parameter.Parameter],
                               optimizers: Union[Optimizer, Sequence[Optimizer]]) -> None:
    r"""Remove ``old_params`` from the ``optimizers`` and insert ``new_params``.

    Newly added parameters will be added to the same :attr:`~torch.optim.Optimizer.param_group` as the removed
    parameters. A :class:`RuntimeError` will be raised if ``old_params`` is split across multiple parameter groups.

    This function differs from :meth:`replace_params_in_optimizer` in that ``len(old_params)`` need not equal
    ``len(new_params)``. However, this function does not support replacing parameters across multiple optimizer
    groups.

    .. warning::

        Dynamically removing parameters from a :class:`~torch.optim.Optimizer` and adding parameters
        to an existing :attr:`~torch.optim.Optimizer.param_group`\s are not officially supported, so this
        function may fail when PyTorch is updated. The
        `recommended practice <https://github.com/pytorch/pytorch/issues/1489#issuecomment-355301737>`_ is
        to instead recreate the optimizer when the parameter set changes
        To simply add new parameters without replacing existing ones, use
        :meth:`~torch.optim.Optimizer.add_param_group`.

    Args:
        old_params (Iterable[torch.nn.parameter.Parameter]):
            Parameters in this iterable should be removed if they are not present in ``new_params``.
        new_params: Parameters in this iterable should be added if they are
            not present in ``old_params``.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer]): One or more
            :class:`~torch.optim.Optimizer` objects

    Raises:
        NotImplementedError: If ``optimizers`` contains more than one optimizer.
        RuntimeError: If not all removed parameters are found in the
            same parameter group, or if any of them are not found at all.
    """
    if len(ensure_tuple(optimizers)) > 1:
        raise NotImplementedError('Surgery with multiple optimizers is not yet supported.')
    opt = ensure_tuple(optimizers)[0]

    # diff the two collection of parameters to find what needs to be removed or added
    # We need to maintain the order of parameters here for training resumption
    # with optimizers that store state so do not use set.
    old_values = list(old_params)
    new_values = list(new_params)
    removed_params = _ordered_diff(old_values, new_values)
    added_params = _ordered_diff(new_values, old_values)

    if len(removed_params) == 0 and len(added_params) == 0:
        return  # nothing to do

    # rip out the removed_params' states from the optimizer
    for p in removed_params:
        if _tensor_in(p, opt.state):  # only true after training starts
            opt.state.pop(p)

    if len(opt.param_groups) == 1:
        group_idx = 0
    else:
        # if there is more than one group, use the ripped out parameters to infer the group
        # to add the new parameters into
        old_group_idxs = [_find_param_in_optimizer(p, opt) for p in removed_params]

        if len(old_group_idxs) == 0:
            raise RuntimeError('No parameters were removed, so unable to infer the group into which to add parameters.')

        missing_param_groups = [x for x in old_group_idxs if x < 0]
        if len(missing_param_groups) > 0:
            raise RuntimeError(f'Parameter groups {missing_param_groups} are not in the optimizer')

        if min(old_group_idxs) != max(old_group_idxs) and len(added_params):
            raise RuntimeError(
                textwrap.dedent("""\
                    Not all removed parameters are in the same parameter group.
                    This makes it unclear where to add the new parameters."""))
        group_idx = old_group_idxs[0]

    param_group = opt.param_groups[group_idx]
    new_param_list = [p for p in param_group['params'] if not _tensor_in(p, removed_params)]
    new_param_list += list(added_params)
    log.debug(f'adding {len(added_params)} new parameters to parameter group #{group_idx}')
    param_group['params'] = new_param_list
