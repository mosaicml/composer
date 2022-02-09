# Copyright 2021 MosaicML. All Rights Reserved.

"""Surgery modifies model architectures.

This module provides helper functions generally used by algorithms that need to modify a provided model, generally by
substituting particular modules for optimized replacements.
"""
import collections
import itertools
import logging
import textwrap
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, OrderedDict, Tuple, Type

from composer.utils.iter_helpers import ensure_tuple

try:
    from typing import Protocol
except ImportError:
    Protocol = object  # Protocol is not available in python 3.7

if TYPE_CHECKING:
    from typing import Protocol

import torch
import torch.distributed

from composer.core.types import Optimizers

log = logging.getLogger(__name__)


class ReplacementFunction(Protocol):
    """Represents a scheme for replacing a model's modules with other modules.

    For typing reasons we represent this as a ``Protocol``, but in practice this class only describes a function.
    Replacement policies return either a replacement module, or None. Return of None means that no modifications will
    be made.

    Args:
        module (torch.nn.Module): Source module
        module_index (int): Optionally used, the i-th instance of module class.

    Returns:
        torch.nn.Module, optional: replacement module, or ``None`` to indicate no modification.
    """

    def __call__(self, module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        ...


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


# adapted from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_module.py#L408
def replace_module_classes(
    module: torch.nn.Module,
    policies: Mapping[Type[torch.nn.Module], ReplacementFunction],
    optimizers: Optional[Optimizers] = None,
    recurse_on_replacements: bool = False,
    indices: Optional[Dict[Any, int]] = None,
) -> Dict[torch.nn.Module, torch.nn.Module]:
    """Modify model in-place by recursively applying replacement policies.

    Examples:
        The following policy::

            policies = {
                nn.Conv2d: lambda x, idx: nn.Linear(16, 32),
                nn.MaxPool2d: lambda x, idx: nn.AvgPool2d(3, stride=2),
                nn.Linear: lambda x, idx: nn.Linear(16, 64) if x.in_features == 32 else None
            }

        will replace all convolution layers with linear layers, and all max pooling with average pooling. Linear
        layers will be optionally replaced depending on the number of input features.

    .. warning:: When a module is replaced, any tensor values within the module are not copied over
                 to the new module even when the shape is identical. For example, if model weights
                 are initialized and prior to calling this function, the initialized weights will not
                 be preserved in any replacements.


    Arguments:
        module (torch.nn.Module): Model to modify.
        policies (Mapping[torch.nn.Module, ReplacementFunction]): Mapping of source class to replacement function. The
            replacement may be either another module or ``None``. If the latter,
            this replacement is skipped.
        recurse_on_replacements (bool): If true, policies will be applied to any module returned
            by another policy. E.g., if one replaces a ``Conv2d`` with a module containing
            another ``Conv2d``, this new child ``Conv2d`` might also be replaced. This can recurse
            infinitely if the replacement policies are not conditioned on
            module properties that change over the course of the recursion.
        indices (Dict[Any, int], optional): A dictionary mapping module types to the number of times
            they've occurred so far in the recursive traversal of
            ``module`` and its child modules. Allows us to pass ``module_index``
            to the replacement policies, so that a policy may switch behavior
            on the i-th instance of the module_class. Note that these indices
            may not correspond to the order in which modules get called in the
            forward pass.
        optimizers (Optimizers, optional): One or more :class:`~torch.optim.Optimizer` objects. If provided,
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
    while len(children_to_parents_and_names) > 0:
        child, parents = children_to_parents_and_names.popitem(last=False)
        for policy_class, replacement_fn in policies.items():
            if not isinstance(child, policy_class):
                continue
            module_index = indices[policy_class]
            replacement = replacement_fn(
                child,
                module_index=module_index,
            )
            indices[policy_class] += 1
            if replacement is not None:
                assert child not in replaced_pairs
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
            f"optimizers was not provided. Be sure to either create the optimizer after invoking this method, or manually add new parameters to the existing optimizer."
        )

    return replaced_pairs


def count_module_instances(model: torch.nn.Module, module_class: Type[torch.nn.Module]) -> int:
    """Counts the number of instances of module_class in the model.

    Example:
        .. testsetup::
            from composer.core.surgery import count_module_instances

        .. doctest::
            >>> from torch import nn
            >>> model = nn.Sequential([nn.Linear(16, 32), nn.Linear(32, 64), nn.ReLU])
            >>> count_module_instances(model, nn.Linear)
            2
            >>> count_module_instances(model, (nn.Linear, nn.ReLU))
            3

    Args:
        model (torch.nn.Module): Source model
        module_class (Type[torch.nn.Module]): module_class to count. Can also be a tuple of classes.

    Returns:
        int: The number of instances of `module_class` in `model`
    """
    count = 0
    for _, child in model.named_children():
        if isinstance(child, module_class):
            count += 1
        count += count_module_instances(child, module_class)

    return count


def _tensor_in(tensor: torch.Tensor, iterable: Iterable[torch.Tensor]):
    """Returns whether `tensor is element` for any element in `iterable` This function is necessary because `tensor in
    iterable` does not work reliably for `Tensor`s.

    See https://discuss.pytorch.org/t/how-to-judge-a-tensor-is-in-a-list/15998/4
    for further discussion.
    """
    return any(tensor is elem for elem in iterable)


def _find_param_in_optimizer(param: torch.nn.parameter.Parameter, optimizer: torch.optim.Optimizer) -> int:
    """Returns the index of the optimizer ``param_group`` containing ``param``
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


def update_params_in_optimizer(old_params: Iterable[torch.nn.parameter.Parameter],
                               new_params: Iterable[torch.nn.parameter.Parameter], optimizers: Optimizers) -> None:
    """Removes old parameters from an optimizer and adds in new parameters Parameters found in `old_params` but not
    `new_params` will be removed from the optimizers.

    Similarly, parameters found in `new_params` but not
    `old_params` will be added to the optimizer. Newly added parameters will
    be added to the same optimizer `param_group` as the removed parameters
    on a best-effort basis. If different removed parameters for a given
    module are in different `param_group`s a RuntimeError will be thrown.
    Dynamically removing parameters from an optimizer and adding parameters
    to an existing `param_group` are not officially supported, so this
    function may fail when PyTorch is updated. The recommended practice is
    to instead recreate the optimizer when the parameter set changes if
    possible. See `recommended practice <https://github.com/pytorch/pytorch/issues/1489#issuecomment-355301737>`_.
    To simply add new parameters without replacing existing ones, use
    :meth:`~torch.optim.Optimizer.add_param_group`.

    Args:
        old_params: Parameters in this iterable should be removed if they are
            not present in `new_params`.
        new_params: Parameters in this iterable should be added if they are
            not present in `old_params`.
        optimizers (Optimizers): One or more `torch.optim.Optimizer` objects

    Raises:
        NotImplementedError: If `optimizers` contains more than one optimizer
        RuntimeError: If not all removed parameters are found in the
            same parameter group, or if any of them are not found at all
    """
    if len(ensure_tuple(optimizers)) > 1:
        raise NotImplementedError("Surgery with multiple optimizers is not yet supported.")
    opt = ensure_tuple(optimizers)[0]

    # diff the two sets of parameters to find what needs to be removed or added
    old_values = set(old_params)
    new_values = set(new_params)
    removed_params = old_values - new_values
    added_params = new_values - old_values

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
            raise RuntimeError("No parameters were removed, so unable to infer the group into which to add parameters.")

        missing_param_groups = [x for x in old_group_idxs if x < 0]
        if len(missing_param_groups) > 0:
            raise RuntimeError(f"Parameter groups {missing_param_groups} are not in the optimizer")

        if min(old_group_idxs) != max(old_group_idxs) and len(added_params):
            raise RuntimeError(
                textwrap.dedent("""\
                    Not all removed parameters are in the same parameter group.
                    This makes it unclear where to add the new parameters."""))
        group_idx = old_group_idxs[0]

    param_group = opt.param_groups[group_idx]
    new_param_list = [p for p in param_group['params'] if not _tensor_in(p, removed_params)]
    new_param_list += list(added_params)
    log.info(f'adding {len(added_params)} new parameters to parameter group #{group_idx}')
    param_group['params'] = new_param_list


def replace_params_in_optimizer(old_params: Iterable[torch.nn.parameter.Parameter],
                                new_params: Iterable[torch.nn.parameter.Parameter], optimizers: Optimizers) -> None:
    """Fully replaces an optimizer's parameters.

    This differs from :meth:`update_params_in_optimizer` in that this method is capable
    of replacing parameters spanning multiple param groups. To accomplish this,
    this function assumes that parameters in ``new_params`` should inherit the
    param group of the corresponding parameter from ``old_params``. Thus, this
    function also assumes that ``old_params`` and ``new_params` have the same length.

    Args:
        old_params (Iterator[torch.nn.parameter.Parameter]): Current parameters of the optimizer.
        new_params (Iterator[torch.nn.parameter.Parameter]): New parameters of the optimizer, given in the same order as
            ``old_params``. Must be the same length as ``old_params``.
        optimizers (Optimizers): One or more :class:`torch.optim.Optimizer` objects.

    Raises:
        NotImplementedError: If ``optimizers`` contains more than one optimizer
        RuntimeError: If ``old_params`` and ``new_params`` have different lengths, or
            if a param from ``old_params`` cannot be found.
    """
    if len(ensure_tuple(optimizers)) > 1:
        raise NotImplementedError("Surgery with multiple optimizers is not yet supported.")

    opt = ensure_tuple(optimizers)[0]
    opt.state.clear()

    param_to_idxs_map = {}
    for group_idx, param_group in enumerate(opt.param_groups):
        param_list = param_group["params"]
        for param_idx, param in enumerate(param_list):
            param_to_idxs_map[param] = (group_idx, param_idx)

    for old_param, new_param in itertools.zip_longest(old_params, new_params):
        if old_params is None or new_params is None:
            raise RuntimeError("old_params and new_params have different lengths.")

        if not old_param in param_to_idxs_map:
            raise RuntimeError(f"Parameter {old_param} is missing from the optimizer.")

        group_idx, param_idx = param_to_idxs_map[old_param]
        opt.param_groups[group_idx]["params"][param_idx] = new_param
