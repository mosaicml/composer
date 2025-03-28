# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
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
    N, hin, num_classes = 4, 2, 2
    x = torch.rand((N, hin))
    y = torch.randint(high=num_classes - 1, size=(N,))
    model = nn.Sequential(nn.Linear(hin, num_classes, bias=False), nn.Softmax(dim=1))
    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for module in model:
        module._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

    model._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
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
            self.mlp = nn.Sequential(
                nn.Linear(num_fmaps, h),
                nn.ReLU(),
                nn.Linear(h, num_classes),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            fmaps = self.conv_model(x)
            vec = torch.mean(fmaps, dim=(2, 3))
            out = self.mlp(vec)
            return out

    # Generate some gradients.
    N, n_ch, num_fmaps, h, num_classes, filter_size = 4, 1, 2, 2, 2, 2
    x = torch.rand((N, n_ch, 8, 8))
    y = torch.randint(high=num_classes - 1, size=(N,))
    model = myNN(n_ch, num_fmaps, h, num_classes, filter_size)

    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for layer in model.modules():
        layer._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

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
        layer._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

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
        tiny_bert_config,
    )  # type: ignore (thirdparty)

    model = HuggingFaceModel(hf_model, metrics=[], use_logits=True)
    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for layer in model.modules():
        layer._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

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
    [simple_model_with_grads, cnn_model_with_grads, simple_transformer_model_with_grads, hf_model_with_grads],
)
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


def _auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if recurse:
        return True

    # With Torch 2.0, there is a bug that emits a nasty warning if you wrap a module with no parameters
    if len(list(module.parameters())) == 0:
        return False

    if hasattr(module, '_fsdp_wrap'):
        return bool(module._fsdp_wrap)
    return False


@pytest.mark.parametrize(
    'model_with_grads',
    [
        simple_model_with_grads,
        cnn_model_with_grads,
        pytest.param(
            simple_transformer_model_with_grads,
            marks=pytest.mark.xfail(reason='SimpleTransformerBase cannot be recursively FSDP wrapped.'),
        ),
        hf_model_with_grads,
    ],
)
@pytest.mark.parametrize('clipping_type', ['norm', 'value'])
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:.*FSDP will not all-gather parameters for containers.*:UserWarning')
@world_size(2)
def test_gradient_clipping_algorithm_with_fsdp_enabled_does_not_error(
    monkeypatch,
    model_with_grads,
    clipping_type,
    dummy_state: State,
    world_size: int,
):
    from torch.distributed.fsdp import FullyShardedDataParallel

    model = model_with_grads()

    clipping_threshold = 0.1191
    state = dummy_state

    state.model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    state.algorithms = [GradientClipping(clipping_type=clipping_type, clipping_threshold=clipping_threshold)]
    logger = Mock()

    engine = Engine(state, logger)
    engine.run_event(Event.AFTER_TRAIN_BATCH)


#### Tests Specific to AGC ######


@pytest.mark.parametrize(
    'model_with_grads',
    [simple_model_with_grads, cnn_model_with_grads, simple_transformer_model_with_grads, hf_model_with_grads],
)
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
    [
        simple_model_with_grads(),
        cnn_model_with_grads(),
        simple_transformer_model_with_grads(),
        hf_model_with_grads(),
    ],
)
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
        weights=weights,
        grad=grad,
        clipping_threshold=clipping_threshold,
    )
    assert torch.equal(clipped_grads, expected)


@pytest.mark.parametrize(
    'weights,grad,expected',
    [
        (torch.Tensor([0., 0.]), torch.Tensor([1., 1.]), torch.Tensor([0., 0.])),
        (torch.Tensor([1., 1.]), torch.Tensor([0., 0.]), torch.Tensor([0., 0.])),
        (torch.Tensor([0., 0.]), torch.Tensor([0., 0.]), torch.Tensor([0., 0.])),
    ],
)
def test_get_clipped_gradients_1D_with_zeros(weights: torch.Tensor, grad: torch.Tensor, expected: torch.Tensor):
    clipping_threshold = 1e-4
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights,
        grad=grad,
        clipping_threshold=clipping_threshold,
    )
    assert torch.equal(clipped_grads, expected)


def test_get_clipped_gradients_2D():
    weights = torch.Tensor([[3., 4.], [9., 40.]])
    grad = torch.Tensor([[7., 24.], [5., 12.]])
    clipping_threshold = 0.5
    expected = torch.Tensor([[0.7, 2.4], [5., 12.]])
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights,
        grad=grad,
        clipping_threshold=clipping_threshold,
    )
    assert torch.equal(clipped_grads, expected)


def test_get_clipped_gradients_3D():

    weights = torch.Tensor([[[3., 8.], [2., 2.]], [[1., 3.], [3., 9.]]])
    grad = torch.Tensor([[[1., 1.], [3., 5.]], [[1., 1.], [1., 1.]]])
    clipping_threshold = 1 / 3.
    expected = torch.Tensor([[[0.5000, 0.5000], [1.5000, 2.5000]], [[1.0000, 1.0000], [1.0000, 1.0000]]])
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights,
        grad=grad,
        clipping_threshold=clipping_threshold,
    )
    assert torch.equal(clipped_grads, expected)


def test_get_clipped_gradients_4D():

    weights = torch.Tensor([[[[3.], [8.]], [[2.], [2.]]], [[[1.], [3.]], [[3.], [9.]]]])
    grad = torch.Tensor([[[[1.], [1.]], [[3.], [5.]]], [[[1.], [1.]], [[1.], [1.]]]])
    clipping_threshold = 1 / 3.
    expected = torch.Tensor([[[[0.5], [0.5]], [[1.5], [2.5]]], [[[1.0], [1.0]], [[1.0], [1.0]]]])
    clipped_grads = grad * _get_clipped_gradient_coeff(
        weights=weights,
        grad=grad,
        clipping_threshold=clipping_threshold,
    )
    assert torch.equal(clipped_grads, expected)
