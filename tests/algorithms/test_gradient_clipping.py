# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from packaging import version
from torch import nn

import composer.algorithms.gradient_clipping.gradient_clipping as gc_module
from composer.algorithms.gradient_clipping import GradientClipping, apply_gradient_clipping
from composer.algorithms.gradient_clipping.gradient_clipping import _apply_agc, _get_clipped_gradient_coeff
from composer.core import Engine, State
from composer.core.event import Event
from tests.common import world_size
from tests.common.datasets import dummy_tiny_bert_classification_batch, dummy_transformer_classifier_batch
from tests.common.models import SimpleTransformerClassifier, configure_tiny_bert_config


def simple_model_with_grads():
    # Set up small NN with one linear layer with no bias + softmax, so only
    # one set of params and get some gradients.
    N, hin, num_classes = 8, 4, 3
    x = torch.rand((N, hin))
    y = torch.randint(high=num_classes - 1, size=(N,))
    model = nn.Sequential(nn.Linear(hin, num_classes, bias=False), nn.Softmax(dim=1))
    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for module in model:
        module._fsdp_wrap = True

    model._fsdp_wrap = True
    o = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    loss.backward()
    return model


def cnn_model_with_grads():
    # Make a CNN with all the common parameters: bias, weight matrix, conv filters.
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

    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for layer in model.modules():
        layer._fsdp_wrap = True

    o = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    loss.backward()
    return model


def simple_transformer_model_with_grads():
    # Make a Transformer model.
    model = SimpleTransformerClassifier(vocab_size=100, num_classes=3)
    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for layer in model.modules():
        layer._fsdp_wrap = True

    x = dummy_transformer_classifier_batch(num_classes=3)
    o = model(x)
    y = torch.randint(high=1, size=o.shape, dtype=o.dtype)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    loss.backward()
    return model


def hf_model_with_grads():
    # Make a HuggingFace BERT model.
    transformers = pytest.importorskip('transformers')

    from composer.models import HuggingFaceModel
    tiny_bert_config = configure_tiny_bert_config()
    tiny_bert_config.num_labels = 3  # type: ignore
    hf_model = transformers.AutoModelForSequenceClassification.from_config(
        tiny_bert_config)  # type: ignore (thirdparty)

    model = HuggingFaceModel(hf_model, metrics=[], use_logits=True)
    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for layer in model.modules():
        layer._fsdp_wrap = True

    x = dummy_tiny_bert_classification_batch(num_classes=3)
    o = model(x).logits
    y = torch.randint(high=1, size=o.shape, dtype=o.dtype)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    loss.backward()
    return model


def test_gradient_clipping_functional(monkeypatch):
    model = Mock()
    new_gc_fn = Mock()
    monkeypatch.setattr(gc_module, '_apply_agc', new_gc_fn)
    apply_gradient_clipping(model, 'adaptive', 0.1, fsdp_enabled=False)
    new_gc_fn.assert_called_once_with(model.parameters(), clipping_threshold=0.1)

    new_gc_fn = Mock()
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', new_gc_fn)
    apply_gradient_clipping(model, 'norm', 0.1, fsdp_enabled=False)
    new_gc_fn.assert_called_once()

    new_gc_fn = Mock()
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_value_', new_gc_fn)
    apply_gradient_clipping(model, 'value', 0.1, fsdp_enabled=False)
    new_gc_fn.assert_called_once()


@pytest.mark.parametrize('clipping_type', [('adaptive',), ('norm',), ('value',)])
@pytest.mark.parametrize(
    'model_with_grads',
    [simple_model_with_grads, cnn_model_with_grads, simple_transformer_model_with_grads, hf_model_with_grads])
def test_gradient_clipping_algorithm(monkeypatch, clipping_type, model_with_grads, dummy_state: State):
    model = model_with_grads()
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


@pytest.mark.parametrize(
    'model_with_grads',
    [simple_model_with_grads(),
     cnn_model_with_grads(),
     simple_transformer_model_with_grads(),
     hf_model_with_grads()])
def test_gradient_clipping_algorithm_with_deepspeed_enabled(
    monkeypatch: pytest.MonkeyPatch,
    model_with_grads,
    dummy_state: State,
):
    clipping_threshold = 0.1191
    apply_gc_fn = Mock()
    monkeypatch.setattr(gc_module, 'apply_gradient_clipping', apply_gc_fn)
    state = dummy_state

    # Set clipping_type to norm to ensure that apply_gradient_clipping
    # is not called.
    state.algorithms = [GradientClipping(clipping_type='norm', clipping_threshold=clipping_threshold)]

    # Enable deepspeed.
    state.deepspeed_config = {}

    model = model_with_grads
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


def _auto_wrap_policy(module: torch.nn.Module, recurse: bool, unwrapped_params: int) -> bool:
    if recurse:
        return True
    if hasattr(module, '_fsdp_wrap'):
        return bool(module._fsdp_wrap)
    return False


@pytest.mark.parametrize('model_with_grads', [
    simple_model_with_grads, cnn_model_with_grads,
    pytest.param(simple_transformer_model_with_grads,
                 marks=pytest.mark.xfail(reason='SimpleTransformerBase cannot be recursively FSDP wrapped.')),
    hf_model_with_grads
])
@pytest.mark.parametrize('clipping_type', ['norm', 'value'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
@pytest.mark.gpu
@world_size(2)
def test_gradient_clipping_algorithm_with_fsdp_enabled_does_not_error(
    monkeypatch,
    model_with_grads,
    clipping_type,
    dummy_state: State,
    world_size: int,
):
    from torch.distributed.fsdp import FullyShardedDataParallel

    clipping_threshold = 0.1191
    state = dummy_state
    state.model = FullyShardedDataParallel(model_with_grads(),
                                           auto_wrap_policy=_auto_wrap_policy,
                                           device_id=torch.cuda.current_device())

    state.algorithms = [GradientClipping(clipping_type=clipping_type, clipping_threshold=clipping_threshold)]
    logger = Mock()

    engine = Engine(state, logger)
    engine.run_event(Event.AFTER_TRAIN_BATCH)


@pytest.mark.parametrize(
    'model_with_grads',
    [simple_model_with_grads, cnn_model_with_grads, simple_transformer_model_with_grads, hf_model_with_grads])
def test_algorithm_with_deepspeed_enabled_errors_out_for_non_norm(
    monkeypatch: pytest.MonkeyPatch,
    dummy_state: State,
    model_with_grads,
):
    clipping_threshold = 0.1191
    apply_gc_fn = Mock()
    monkeypatch.setattr(gc_module, 'apply_gradient_clipping', apply_gc_fn)
    state = dummy_state

    # Enable deepspeed and set clipping_type to norm to ensure that apply_gradient_clipping
    # is not called.
    state.algorithms = [GradientClipping(clipping_type='value', clipping_threshold=clipping_threshold)]
    state.deepspeed_config = {}

    model = model_with_grads()
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


@pytest.mark.parametrize(
    'model_with_grads',
    [simple_model_with_grads, cnn_model_with_grads, simple_transformer_model_with_grads, hf_model_with_grads])
def test_apply_agc(model_with_grads):

    model = model_with_grads()
    # Make sure after calling apply_agc, the gradients inside the model are
    # the same as if we manually called _get_clipped_gradients on the weights and
    # gradients.
    weights = next(model.parameters())
    grad = weights.grad
    expected_clipped_grad = grad.detach() * _get_clipped_gradient_coeff(weights, grad)
    _apply_agc(model.parameters(), 0.01)
    current_grad = next(model.parameters()).grad
    torch.equal(current_grad, expected_clipped_grad)


@pytest.mark.parametrize(
    'model_with_grads',
    [simple_model_with_grads(),
     cnn_model_with_grads(),
     simple_transformer_model_with_grads(),
     hf_model_with_grads()])
def test_apply_agc_does_not_error(model_with_grads):
    """This test is just to ensure that no errors are raised.

    Accuracy of the AGC calculations are tested in other tests.
    """

    model = model_with_grads
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
