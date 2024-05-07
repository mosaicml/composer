from composer.checkpoint.state_dict import get_model_state_dict
from tests.common.models import EvenSimplerMLP
import torch 
import pytest
import fnmatch
#TODO write unit tests


def test_get_model_state_dict_full():
    model = EvenSimplerMLP(num_features=8, device='cpu')
    model_state_dict = get_model_state_dict(model, sharded=False, precision=None, include_keys=None, ignore_keys=None)
    for name, param in model.named_parameters():
        print(name)
        assert name in model_state_dict
        assert torch.equal(model_state_dict[name], param)


def test_get_model_state_dict_include():
    model = EvenSimplerMLP(num_features=8, device='cpu')
    model_state_dict = get_model_state_dict(model, 
                                            sharded=False,
                                            precision=None,
                                            include_keys=['net.0.weight'])
    assert set(model_state_dict.keys()) == {'net.0.weight'}

    model_state_dict = get_model_state_dict(model, 
                                            sharded=False,
                                            precision=None,
                                            include_keys='net.1*')
    assert set(model_state_dict.keys()) == {'net.1.weight'}
    
def test_get_model_state_dict_ignore():
    model = EvenSimplerMLP(num_features=8, device='cpu')

    model_state_dict = get_model_state_dict(model, 
                                            sharded=False,
                                            precision=None,
                                            ignore_keys='net.1.weight')
    assert set(model_state_dict.keys()) == {'net.0.weight'}

    model_state_dict = get_model_state_dict(model, 
                                            sharded=False,
                                            precision=None,
                                            ignore_keys=['net.1*'])
    assert set(model_state_dict.keys()) == {'net.0.weight'}


#TODO add tests for sharded and for precision