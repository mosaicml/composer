from typing import Sequence, Mapping

__all__ = ['extract_hparams', 'convert_nested_dict_to_flat_dict', 'convert_flat_dict_to_nested_dict']

def _parse_dict_for_hparams(name, dic):
    '''Grabs hyperparamters from each element in dictionary and optionally names dictionary'''
    hparams_to_add = {}
    for k, v in dic.items():
            hparams_to_add[k] = _grab_hparams(v)
    # Wrap dictionary in parent dictionary and add name
    if name is not None:
        hparams_to_add = {name: hparams_to_add}
    return hparams_to_add
    

def _grab_hparams(obj):
    """Helper function that recursively parses objects for their hyparameters."""
    
    # If the object has already grabbed its hyperparameters (its a Composer object)
    # then parse hparams attribute (which is a dict) and name those sub-hyperparameters
    if hasattr(obj, 'hparams'):
        obj_name = obj.__class__.__name__
        hparams_to_add = _parse_dict_for_hparams(name=obj_name, dic=obj.hparams)
        
    # If object has a __dict__ atrribute then parse all its members as hparams udner the name obj.__class__.__name__
    elif hasattr(obj, '__dict__'):
        obj_name = obj.__class__.__name__
        hparams_to_add = _parse_dict_for_hparams(name=obj.__class__.__name__, dic=vars(obj))
        
    # If object is a dict or mapping object then parse all elements with no parent name.
    elif isinstance(obj, Mapping):
        hparams_to_add = _parse_dict_for_hparams(name=None, dic=obj)
        
    # If object is sequence then loop through a parse each element in sequence
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        hparams_to_add = {}
        for sub_obj in obj:
            sub_obj_dic = _grab_hparams(sub_obj)
            hparams_to_add.update(sub_obj_dic)
            
    # Otherwise the object is a primitive type like int, str, etc.
    else:
        if obj.__class__.__module__ == 'builtins':
            hparams_to_add = obj
        else:
            hparams_to_add = {obj.__class__.__name__: obj}
    return hparams_to_add
    

def extract_hparams(locals_dict):
    """Takes in local symbol table and parses"""
    hparams = {}
    try:
        # remove self from locals()
        locals_dict.pop('self')
    except KeyError:
        pass
    for k,v in locals_dict.items():
        hparams_to_add = _grab_hparams(v)

        hparams[k] = hparams_to_add
    return hparams


def convert_nested_dict_to_flat_dict(prefix, my_dict):
    prefix = prefix + '/' if prefix else ''
    d = {}
    for k,v in my_dict.items():
        if isinstance(v, dict):
            flat_dict = convert_nested_dict_to_flat_dict(prefix=prefix + k,my_dict=v)
            d.update(flat_dict)
        else:
            d[prefix + k] = v
    return d


def convert_flat_dict_to_nested_dict(flat_dict):
    nested_dic = {}
    for k,v in flat_dict.items():
        cur_dict = nested_dic
        sub_keys = k.split('/')
        for sub_key in sub_keys[:-1]:
            if sub_key not in cur_dict:
                cur_dict[sub_key] = {}
            cur_dict = cur_dict[sub_key]
        cur_dict[sub_keys[-1]] = v
    return nested_dic