# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import composer.algorithms.gradient_clipping.gradient_clipping as gc_module
from composer.algorithms.gradient_clipping import GradientClipping, apply_gradient_clipping
from composer.algorithms.gradient_clipping.gradient_clipping import _apply_agc, _get_clipped_gradient_coeff
from composer.core import Engine
from composer.core.event import Event
from tests.fixtures import dummy_fixtures

# To satisfy pyright.
dummy_state = dummy_fixtures.dummy_state


@pytest.fixture
def simple_model_with_grads():
    # Set up small NN with one linear layer with no bias + softmax, so only
    # one set of params and get some gradients.
    N, hin, num_classes = 8, 4, 3
    x = torch.rand((N, hin))
    y = torch.randint(high=num_classes - 1, size=(N,))
    model = nn.Sequential(nn.Linear(hin, num_classes, bias=False), nn.Softmax(dim=1))
    o = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    loss.backward()
    return model


@pytest.fixture
def cnn_model_with_grads():
    # Make a NN with all the common parameters: bias, weight matrix, conv filters.
    class myNN(nn.Module):

        def __init__(self, n_ch, num_fmaps, h, num_classes, filter_size):
            super().__init__()
            self.conv_model = nn.Sequential(nn.Conv2d(n_ch, num_fmaps, kernel_size=filter_size), nn.ReLU())
            self.mlp = nn.Sequential(nn.Linear(num_fmaps, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                     nn.Linear(h, num_classes), nn.Softmax(dim=1))

        def forward(self, x):
            fmaps = self.conv_model(x)
            vec = torch.mean(fmaps, dim=(2, 3))
            out = self.mlp(vec)
            return out

    # Generate some gradients.
    N, n_ch, num_fmaps, h, num_classes, filter_size = 8, 3, 4, 4, 3, 3
    x = torch.rand((N, n_ch, 16, 16))
    y = torch.randint(high=num_classes - 1, size=(N,))
    model = myNN(n_ch, num_fmaps, h, num_classes, filter_size)
    o = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    loss.backward()
    return model


def test_gradient_clipping_functional(monkeypatch):
    parameters = Mock()
    new_gc_fn = Mock()
    monkeypatch.setattr(gc_module, '_apply_agc', new_gc_fn)
    apply_gradient_clipping(parameters, 'adaptive', 0.1)
    new_gc_fn.assert_called_once_with(parameters, clipping_threshold=0.1)

    new_gc_fn = Mock()
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', new_gc_fn)
    apply_gradient_clipping(parameters, 'norm', 0.1)
    new_gc_fn.assert_called_once()

    new_gc_fn = Mock()
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_value_', new_gc_fn)
    apply_gradient_clipping(parameters, 'value', 0.1)
    new_gc_fn.assert_called_once()


@pytest.mark.parametrize('clipping_type', [('adaptive',), ('norm',), ('value',)])
def test_gradient_clipping_algorithm(monkeypatch, clipping_type, simple_model_with_grads, dummy_state):
    model = simple_model_with_grads
    apply_gc_fn = Mock()
    monkeypatch.setattr(gc_module, 'apply_gradient_clipping', apply_gc_fn)
    state = dummy_state
    state.model = model
    state.callbacks = []
    state.algorithms = [GradientClipping(clipping_type=clipping_type, clipping_threshold=0.01)]
    logger = Mock()
    engine = Engine(state, logger)

    # Run the Event that should cause gradient_clipping.apply to be called.
    engine.run_event(Event.AFTER_TRAIN_BATCH)

    apply_gc_fn.assert_called_once()


def test_gradient_clipping_algorithm_with_deepspeed_enabled(monkeypatch: pytest.MonkeyPatch, simple_model_with_grads,
                                                            dummy_state):
    clipping_threshold = 0.1191
    apply_gc_fn = Mock()
    monkeypatch.setattr(gc_module, 'apply_gradient_clipping', apply_gc_fn)
    state = dummy_state

    # Set clipping_type to norm to ensure that apply_gradient_clipping
    # is not called.
    state.algorithms = [GradientClipping(clipping_type='norm', clipping_threshold=clipping_threshold)]

    # Enable deepspeed.
    state.deepspeed_config = {}

    model = simple_model_with_grads
    state.model = model
    logger = Mock()
    engine = Engine(state, logger)

    # Run the Event that should cause gradient_clipping.apply to be called and deepspeed_config to be modified.
    engine.run_event(Event.INIT)

    # Make sure deepspeed_config's gradient_clipping field is set properly.
    assert 'gradient_clipping' in state.deepspeed_config and state.deepspeed_config[
        'gradient_clipping'] == clipping_threshold

    # Make sure apply_gradient_clipping is not called.
    apply_gc_fn.assert_not_called()


def test_algorithm_with_deepspeed_enabled_errors_out_for_non_norm(monkeypatch: pytest.MonkeyPatch, dummy_state):
    clipping_threshold = 0.1191
    apply_gc_fn = Mock()
    monkeypatch.setattr(gc_module, 'apply_gradient_clipping', apply_gc_fn)
    state = dummy_state

    # Enable deepspeed and set clipping_type to norm to ensure that apply_gradient_clipping
    # is not called.
    state.algorithms = [GradientClipping(clipping_type='value', clipping_threshold=clipping_threshold)]
    state.deepspeed_config = {}

    model = simple_model_with_grads
    state.model = model
    logger = Mock()
    engine = Engine(state, logger)

    # Clipping type is not set to norm and deepspeed is enabled so NotImplementedError should be raised.
    with pytest.raises(NotImplementedError):
        engine.run_event(Event.INIT)

    # Clipping threshold is less than zero and deepspeed is enabled so NotImplementedError should be raised.
    state.algorithms = [GradientClipping(clipping_type='norm', clipping_threshold=-2.0)]
    with pytest.raises(ValueError):
        engine.run_event(Event.INIT)


#### Tests Specific to AGC ######


def test_apply_agc(simple_model_with_grads):

    model = simple_model_with_grads
    # Make sure after calling apply_agc, the gradients inside the model are
    # the same as if we manually called _get_clipped_gradients on the weights and
    # gradients.
    weights = next(model.parameters())
    grad = weights.grad
    expected_clipped_grad = grad.detach() * _get_clipped_gradient_coeff(weights, grad)
    _apply_agc(model.parameters(), 0.01)
    current_grad = next(model.parameters()).grad
    torch.equal(current_grad, expected_clipped_grad)


def test_apply_agc_with_cnn_does_not_error(cnn_model_with_grads):
    """This test is just to ensure that no errors are raised.

    Accuracy of the AGC calculations are tested in other tests.
    """

    model = cnn_model_with_grads
    # Call apply_agc. If this function returns then we know that nothing errored out.
    _apply_agc(model.parameters(), 0.01)


def test_get_clipped_gradients_1D():
    weights = torch.Tensor([3., 4.])
    grad = torch.Tensor([7., 24.])
    clipping_threshold = 0.5
    expected = torch.Tensor([0.7, 2.4])
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights, grad=grad, clipping_threshold=clipping_threshold)
    assert torch.equal(clipped_grads, expected)


@pytest.mark.parametrize('weights,grad,expected',
                         [(torch.Tensor([0., 0.]), torch.Tensor([1., 1.]), torch.Tensor([0., 0.])),
                          (torch.Tensor([1., 1.]), torch.Tensor([0., 0.]), torch.Tensor([0., 0.])),
                          (torch.Tensor([0., 0.]), torch.Tensor([0., 0.]), torch.Tensor([0., 0.]))])
def test_get_clipped_gradients_1D_with_zeros(weights: torch.Tensor, grad: torch.Tensor, expected: torch.Tensor):
    clipping_threshold = 1e-4
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights, grad=grad, clipping_threshold=clipping_threshold)
    assert torch.equal(clipped_grads, expected)


def test_get_clipped_gradients_2D():
    weights = torch.Tensor([[3., 4.], [9., 40.]])
    grad = torch.Tensor([[7., 24.], [5., 12.]])
    clipping_threshold = 0.5
    expected = torch.Tensor([[0.7, 2.4], [5., 12.]])
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights, grad=grad, clipping_threshold=clipping_threshold)
    assert torch.equal(clipped_grads, expected)


def test_get_clipped_gradients_3D():

    weights = torch.Tensor([[[3., 8.], [2., 2.]], [[1., 3.], [3., 9.]]])
    grad = torch.Tensor([[[1., 1.], [3., 5.]], [[1., 1.], [1., 1.]]])
    clipping_threshold = 1 / 3.
    expected = torch.Tensor([[[0.5000, 0.5000], [1.5000, 2.5000]], [[1.0000, 1.0000], [1.0000, 1.0000]]])
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights, grad=grad, clipping_threshold=clipping_threshold)
    assert torch.equal(clipped_grads, expected)


def test_get_clipped_gradients_4D():

    weights = torch.Tensor([[[[3.], [8.]], [[2.], [2.]]], [[[1.], [3.]], [[3.], [9.]]]])
    grad = torch.Tensor([[[[1.], [1.]], [[3.], [5.]]], [[[1.], [1.]], [[1.], [1.]]]])
    clipping_threshold = 1 / 3.
    expected = torch.Tensor([[[[0.5], [0.5]], [[1.5], [2.5]]], [[[1.0], [1.0]], [[1.0], [1.0]]]])
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights, grad=grad, clipping_threshold=clipping_threshold)
    assert torch.equal(clipped_grads, expected)
