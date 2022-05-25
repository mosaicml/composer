# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import composer.functional as cf
from composer.algorithms.agc import AGC
from composer.algorithms.agc.agc import _get_clipped_gradient_coeff
from composer.core import Engine
from composer.core.event import Event


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


def test_agc_functional(simple_model_with_grads):

    model = simple_model_with_grads
    # Make sure after calling apply_agc, the gradients inside the model are
    # the same as if we manually called _get_clipped_gradients on the weights and
    # gradients.
    weights = next(model.parameters())
    grad = weights.grad
    expected_clipped_grad = grad.detach() * _get_clipped_gradient_coeff(weights, grad)
    cf.apply_agc(model)
    current_grad = next(model.parameters()).grad
    torch.equal(current_grad, expected_clipped_grad)


def test_agc_functional_with_cnn_does_not_error(cnn_model_with_grads):
    """This test is just to ensure that no errors are raised.

    Accuracy of the AGC calculations are tested in other tests.
    """

    model = cnn_model_with_grads
    # Call apply_agc. If this function returns then we know that nothing errored out.
    cf.apply_agc(model)


def test_AGC_algorithm(simple_model_with_grads):
    # Get weight and gradients from the fixture model.
    model = simple_model_with_grads
    weights = next(model.parameters())
    grad = weights.grad
    expected_clipped_grad = grad.detach() * _get_clipped_gradient_coeff(weights, grad)

    # Set up a mock engine.
    state = Mock()
    state.model = model
    state.profiler.marker = Mock(return_value=None)
    state.callbacks = []
    state.algorithms = [AGC()]
    logger = Mock()
    engine = Engine(state, logger)

    # Run the Event that should cause AGC.apply to be called.
    engine.run_event(Event.AFTER_TRAIN_BATCH)

    # Check that the gradients weights holds are equivalent to the calling
    # grad.detach() * _get_clipped_gradient_coeff on the weights and grads.
    current_grad = next(model.parameters()).grad
    torch.equal(current_grad, expected_clipped_grad)


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
