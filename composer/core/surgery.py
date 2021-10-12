import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Type

import torch

log = logging.getLogger(__name__)


class ReplacementFunction(Protocol):
    """
    For typing, we define a ``ReplacementFunction`` to represent replacement policies. These policies
    return either a replacement module, or None. Return of None means the no modifications will be made.

    Replacement policies return either a replacement module, or None. Return of None
    means the no modifications will be made.

    Args:
        module: source module
        module_index: optionally used, the i-th instance of module class

    Returns replacement module ``torch.nn.Module`` or ``None``.
    """

    def __call__(self, module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        ...


# adapted from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_module.py#L408
def replace_module_classes(
    model: torch.nn.Module,
    policies: Dict[Any, ReplacementFunction],
    recurse_on_replacements: bool = False,
    indices: Optional[Dict[Any, int]] = None,
) -> List[Tuple[torch.nn.Module, torch.nn.Module]]:
    """ Modify model in-place by recursively applying replacement policies. Replacement policies are a mapping
    of source classes and `ReplacementFunction`.

    Examples:
        The following policy::

            policies = {
                nn.Conv2d: lambda x, idx: nn.Linear(16, 32),
                nn.MaxPool2d: lambda x, idx: nn.AvgPool2d(3, stride=2),
                nn.Linear: lambda x, idx: nn.Linear(16, 64) if x.in_features == 32 else None
            }

        will replace all convolution layers with linear layers, and all max pooling with average pooling. Linear
        layers will be optionally replaced depending on the number of input features.


    Arguments:
        module: Model to modify.
        policies: Mapping of source class to replacement function. The
            replacement may be either another module or `None`. If the latter,
            this replacement is skipped.
        recurse_on_replacements: If true, policies will be applied to any module returned
            by another policy. E.g., if one replaces a `Conv2d` with a module containing
            another `Conv2d`, this new child `Conv2d` might also be replaced. This can recurse
            infinitely if the replacement policies are not conditioned on
            module properties that change over the course of the recursion.
        indices: A dictionary mapping module types to the number of times
            they've occurred so far in the recursive traversal of
            `model` and its child modules. Allows us to pass `module_index`
            to the replacement policies, so that a policy may switch behavior
            on the i-th instance of the module_class. Note that these indices
            may not correspond to the order in which modules get called in the
            forward pass.

    Returns:
        replaced_pairs: a list of pairs of
            (original module, replacement module), reflecting the replacements
            applied to `module` and its children.

    """
    replaced_pairs = []
    indices = indices if indices is not None else {c: 0 for c in policies}
    for name, child in model.named_children():
        already_recursed = False
        child_class = child.__class__
        if child_class in policies:
            module_index = indices[child_class]
            replacement = policies[child_class](
                child,
                module_index=module_index,
            )
            indices[child_class] += 1
            if replacement is not None:
                replaced_pairs.append((child, replacement))
                if recurse_on_replacements:
                    # recurse on new child object
                    replaced_pairs += replace_module_classes(
                        replacement,
                        policies,
                        recurse_on_replacements=recurse_on_replacements,
                        indices=indices,
                    )
                already_recursed = True
                setattr(model, name, replacement)

        if not already_recursed:
            replaced_pairs += replace_module_classes(
                child,
                policies,
                recurse_on_replacements=recurse_on_replacements,
                indices=indices,
            )

    return replaced_pairs


def tensor_in(tensor: torch.Tensor, iterable: Iterable[torch.Tensor]):
    """Returns whether `tensor is element` for any element in `iterable`

    This function is necessary because `tensor in iterable` does not work
    reliably for `Tensor`s.

    See https://discuss.pytorch.org/t/how-to-judge-a-tensor-is-in-a-list/15998/4
    for further discussion.
    """
    return any(tensor is elem for elem in iterable)


def find_param_in_optimizer(param: torch.Tensor, opt: torch.optim.Optimizer) -> int:
    """Returns the index of the optimizer `param_group` containing `param`

    Optimizers store their parameters within an iterable of `dict`s called
    `param_groups`. By default, there is only one group in `param_groups`
    that containing all the parameters, but there can be more than one. This
    function is a simple utility to identify which parameter group in
    `param_groups` contains a given parameter, if any. The information
    might be desirable to, e.g., inspect the optimizer settings being used
    for a given parameter, or to remove unused parameter tensors from
    the optimizer.

    Args:
        param: `Parameter` to search for
        opt: `Optimizer` to search within

    Returns:
        The index within `opt.param_groups` of the first group containing
        param. If not found, returns -1.
    """
    for i, group in enumerate(opt.param_groups):
        param_list: List[torch.Tensor] = group['params']
        if tensor_in(param, param_list):
            return i
    return -1


def count_module_instances(model: torch.nn.Module, module_class: Type[torch.nn.Module]) -> int:
    """
    Counts the number of instances of module_class in the model.

    Example:
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
