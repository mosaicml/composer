# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type

import torch

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
