# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Mapping, Sequence, Dict, Any

import torch

__all__ = ['extract_hparams', 'convert_nested_dict_to_flat_dict', 'convert_flat_dict_to_nested_dict']


def _parse_dict_for_hparams(dic: Dict) -> Dict:
    '''Grabs hyperparamters from each element in dictionary and optionally names dictionary'''
    hparams_to_add = {}
    for k, v in dic.items():
        if k.startswith('_') or isinstance(v, torch.Tensor) or callable(v) or k == 'defaults' or v is None:
            continue
        hparams_to_add[k] = _grab_hparams(v)
    return hparams_to_add


def _grab_hparams(obj):
    """Helper function that recursively parses objects for their hyparameters."""

    # If the object has already grabbed its hyperparameters (its a Composer object)
    # then parse hparams attribute (which is a dict) and name those sub-hyperparameters
    if hasattr(obj, 'local_hparams'):
        obj_name = obj.__class__.__name__
        parsed_hparams = _parse_dict_for_hparams(dic=obj.local_hparams)
        hparams_to_add = {obj_name: parsed_hparams}

    # If object has a __dict__ atrribute then parse all its members as hparams udner the name obj.__class__.__name__
    elif hasattr(obj, '__dict__'):
        obj_name = obj.__class__.__name__
        parsed_hparams = _parse_dict_for_hparams(dic=vars(obj))
        # sig = inspect.signature(obj.__class__.__init__)
        # defaults = {k: v.default for k, v in sig.parameters.items() if k != 'self'}
        # parsed_hparams = {k: v for k, v in parsed_hparams.items() if not (k in defaults and defaults[k] == v)}
        hparams_to_add = {obj_name: parsed_hparams}

    # If object is a dict or mapping object then parse all elements with no parent name.
    elif isinstance(obj, Mapping):
        hparams_to_add = _parse_dict_for_hparams(dic=obj)

    # If object is sequence then loop through a parse each element in sequence
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        hparams_to_add = {}
        hparams_to_add_seq = []
        for sub_obj in obj:
            parsed_sub_obj = _grab_hparams(sub_obj)
            if isinstance(parsed_sub_obj, dict):
                hparams_to_add.update(parsed_sub_obj)
            else:
                hparams_to_add_seq.append(parsed_sub_obj)
        if hparams_to_add_seq:
            if not hparams_to_add:
                hparams_to_add = hparams_to_add_seq
            else:
                hparams_to_add['seq'] = hparams_to_add_seq

    # Otherwise the object is a primitive type like int, str, etc.
    else:
        if obj.__class__.__module__ == 'builtins':
            if obj is None:
                obj = 'None'
            hparams_to_add = obj
        else:
            hparams_to_add = {obj.__class__.__name__: obj}
    return hparams_to_add


def extract_hparams(locals_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Takes in local symbol table and parses"""
    hparams = {}
    try:
        # remove self from locals()
        locals_dict.pop('self')
    except KeyError:
        pass
    for k, v in locals_dict.items():
        hparams_to_add = _grab_hparams(v)

        hparams[k] = hparams_to_add
    return hparams


def convert_nested_dict_to_flat_dict(my_dict, prefix=None):
    prefix = prefix + '/' if prefix else ''
    d = {}
    for k, v in my_dict.items():
        if isinstance(v, dict):
            flat_dict = convert_nested_dict_to_flat_dict(prefix=prefix + k, my_dict=v)
            d.update(flat_dict)
        else:
            d[prefix + k] = v
    return d


def convert_flat_dict_to_nested_dict(flat_dict):
    nested_dic = {}
    for k, v in flat_dict.items():
        cur_dict = nested_dic
        sub_keys = k.split('/')
        for sub_key in sub_keys[:-1]:
            if sub_key not in cur_dict:
                cur_dict[sub_key] = {}
            cur_dict = cur_dict[sub_key]
        cur_dict[sub_keys[-1]] = v
    return nested_dic
