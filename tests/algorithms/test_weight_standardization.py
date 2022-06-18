from tests.common import SimpleConvModel
import pytest
import torch
from composer.algorithms.weight_standardization.weight_standardization import apply_weight_standardization

@pytest.fixture
def sample_conv_model():
    return SimpleConvModel()


def test_weight_standardization(sample_conv_model):
    sc = sample_conv_model
    apply_weight_standardization(sc)
    assert torch.allclose(torch.zeros_like(sc.conv1.weight[:,0, 0 ,0]), sc.conv1.weight.mean([1,2,3]), atol=1e-7)
    assert torch.allclose(torch.ones_like(sc.conv1.weight[:, 0, 0 ,0]), sc.conv1.weight.std([1,2,3]))
    assert torch.allclose(torch.zeros_like(sc.conv2.weight[:,0, 0 ,0]), sc.conv2.weight.mean([1,2,3]), atol=1e-7)
    assert torch.allclose(torch.ones_like(sc.conv2.weight[:, 0, 0 ,0]), sc.conv2.weight.std([1,2,3]))




