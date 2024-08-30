# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Contains helper functions for auto-logging hparams."""

from enum import Enum
from typing import Any

__all__ = ['extract_hparams', 'convert_nested_dict_to_flat_dict', 'convert_flat_dict_to_nested_dict']


def extract_hparams(locals_dict: dict[str, Any]) -> dict[str, Any]:
    """Takes in local symbol table and recursively grabs any hyperparameter.

    Args:
        locals_dict (dict[str, Any]): The local symbol table returned when calling locals(),
            which maps any free local variables' names to their values.

    Returns:
        dict[str, Any]: A nested dictionary with every element of locals_dict mapped to its
            value or to another sub_dict.
    """
    hparams = {}
    for k, v in locals_dict.items():
        if k.startswith('_') or k == 'self' or type(v) is type:
            continue
        hparams_to_add = _grab_hparams(v)
        hparams[k] = hparams_to_add
    return hparams


def _grab_hparams(obj) -> Any:
    """Helper function parses objects for their hyperparameters going only one level deep."""
    # If the object has already grabbed its hyperparameters (it calls extract_hparams inside __init__)
    # then parse hparams attribute (which is a dict) and name those sub-hyperparameters
    if hasattr(obj, 'local_hparams'):
        return {obj.__class__.__name__: obj.local_hparams}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_get_obj_repr(sub_obj) for sub_obj in obj]
    elif isinstance(obj, dict):
        return {k: _get_obj_repr(sub_obj) for k, sub_obj in obj.items()}
    else:
        return _get_obj_repr(obj)


def _get_obj_repr(obj: Any):
    """Returns best representation of object.

    Args:
        obj (Any): the object.

    Returns:
        obj if obj is None or it is a int, float, str, bool type.
        obj.value if obj is an Enum. Otherwise returns obj.__class__.__name__.
    """
    if any(isinstance(obj, type_) for type_ in [int, float, str, bool]) or obj is None:
        return obj
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj.__class__.__name__


def convert_nested_dict_to_flat_dict(nested_dict: dict, prefix='') -> dict:
    """Takes in a nested dict converts it to a flat dict with keys separated by slashes.

    Args:
        nested_dict (dict): A dictionary containing at least one other dictionary.
        prefix (str, optional): A prefix to left append to the keys in the dictionary.
            'Defaults to ''.

    Returns:
        dict: A flat dictionary representation of the nested one (contains no other
            dictionaries inside of it)
    """
    flat_dict = {}
    for k, v in nested_dict.items():
        key = prefix + '/' + k if prefix != '' else k
        # Recursively crawl sub-dictionary.
        if isinstance(v, dict):
            sub_flat_dict = convert_nested_dict_to_flat_dict(prefix=key, nested_dict=v)
            flat_dict.update(sub_flat_dict)
        else:
            flat_dict[key] = v
    return flat_dict


def convert_flat_dict_to_nested_dict(flat_dict: dict) -> dict:
    """Converts flat dictionary separated by slashes to nested dictionary.

    Args:
        flat_dict (dict): flat dictionary containing no sub-dictionary with keys
            separated by slashes. e.g. {'a':1, 'b/c':2}

    Returns:
        dict: a nested dict.
    """
    nested_dict = {}
    for k, v in flat_dict.items():
        # Initially sub_dict is the main nested_dict, but we will continually update it to be the
        # sub-dictionary of sub_dict.
        sub_dict = nested_dict
        sub_keys = k.split('/')
        for sub_key in sub_keys[:-1]:
            if sub_key not in sub_dict:
                # Create a new sub-dictionary inside of sub_dict.
                sub_dict[sub_key] = {}
            # Change the sub_dict reference to be the sub-dictionary of sub_dict (i.e. go one level deeper).
            sub_dict = sub_dict[sub_key]
        # The last key in sub_keys does not map to a dict. It just maps to v.
        sub_dict[sub_keys[-1]] = v
    # Changes to sub_dict will be reflected in nested_dict, so we can just return nested_dict.
    return nested_dict
