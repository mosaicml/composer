# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Mapping, Sequence, Dict, Any
import array
import numpy as np
import torch

__all__ = ['extract_hparams', 'convert_nested_dict_to_flat_dict', 'convert_flat_dict_to_nested_dict']


def _parse_dict_for_hparams(dic: Dict) -> Dict:
    '''Grabs hyperparamters from each element in dictionary.'''
    hparams_to_add = {}
    for k, v in dic.items():
        if k.startswith('_') or k == 'defaults':
            continue
        hparams = _grab_hparams(v)
        if hparams == {}:
            continue
        hparams_to_add[k] = _grab_hparams(v)
    return hparams_to_add


def _grab_hparams(obj):
    """Helper function that recursively parses objects for their hyperparameters."""

    if any([isinstance(obj, torch.Tensor), isinstance(obj, np.ndarray), isinstance(obj, array.array), callable(obj)]):
        return {}

    # If the object has already grabbed its hyperparameters (it calls extract_hparams inside __init__)
    # then parse hparams attribute (which is a dict) and name those sub-hyperparameters
    if hasattr(obj, 'local_hparams'):
        obj_name = obj.__class__.__name__
        parsed_hparams = _parse_dict_for_hparams(dic=obj.local_hparams)
        hparams_to_add = {obj_name: parsed_hparams}
        return hparams_to_add

    # If object has a __dict__ atrribute then parse all its members as hparams under the name obj.__class__.__name__
    elif hasattr(obj, '__dict__'):
        obj_name = obj.__class__.__name__
        parsed_hparams = _parse_dict_for_hparams(dic=vars(obj))
        hparams_to_add = {obj_name: parsed_hparams}
        return hparams_to_add

    # If object is a dict or mapping object then parse all elements with no parent name.
    elif isinstance(obj, Mapping):
        hparams_to_add = _parse_dict_for_hparams(dic=obj)
        return hparams_to_add

    # If object is sequence then loop through a parse each element in sequence
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        hparams_to_add_dict = {}
        hparams_to_add_seq = []
        for sub_obj in obj:
            parsed_sub_obj = _grab_hparams(sub_obj)
            if parsed_sub_obj is not None:
                if isinstance(parsed_sub_obj, dict):
                    hparams_to_add_dict.update(parsed_sub_obj)
                else:
                    hparams_to_add_seq.append(parsed_sub_obj)
        if hparams_to_add_seq:
            
            if len(hparams_to_add_dict) == 0:
                return hparams_to_add_seq
            else:
                hparams_to_add_dict['seq'] = hparams_to_add_seq
        return hparams_to_add_dict
        
    # The object is an instance of a class that does not have a __dict__ or local_hparams members.
    else:
        # If the object is a primitive type like int, str, etc. then return not inside a dict
        if obj.__class__.__module__ == 'builtins':
            if obj is None:
                obj = 'None'
            return obj

        # The object is some object of a custom class, so just return it with the class name.
        else:
            return {obj.__class__.__name__: obj}


def extract_hparams(locals_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Takes in local symbol table and recursively grabs any hyperparameter."""
    hparams = {}
    if 'self' in locals_dict:
        locals_dict.pop('self')
    try:
        for k, v in locals_dict.items():
            hparams_to_add = _grab_hparams(v)
            hparams[k] = hparams_to_add
    except RecursionError as e:
        raise RecursionError(f'Crawled too deeply into Trainer members and members of members, etc. during auto-logging of hparams: {str(e)}')
    return hparams


def convert_nested_dict_to_flat_dict(nested_dict, prefix=''):
    flat_dict = {}
    for k, v in nested_dict.items():
        key = prefix + '/' + k if prefix != '' else k
        # Recursively crawl sub-dictionary.
        if isinstance(v, dict):
            flat_dict = convert_nested_dict_to_flat_dict(prefix=key, nested_dict=v)
            flat_dict.update(flat_dict)
        else:
            flat_dict[key] = v
    return flat_dict


def convert_flat_dict_to_nested_dict(flat_dict):
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
