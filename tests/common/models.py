# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Contains commonly used models that are shared across the test suite."""
import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Union

import pytest
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import resnet

from composer.loss import loss_registry
from composer.metrics import CrossEntropy, MIoU
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import ComposerClassifier, HuggingFaceModel, Initializer

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class EmptyModel(ComposerClassifier):
    """Always predict 0 with no parameters."""

    def __init__(self, num_classes: int = 2):
        super().__init__(module=torch.nn.Sequential(), num_classes=num_classes)

    def forward(self, x):
        out = torch.rand([x[1].shape[0], 2], dtype=x[0].dtype)
        out[:, 0] = 0.99
        out[:, 1] = 0.01
        return out


class ZeroModel(ComposerClassifier):
    """Always predict 0."""

    def __init__(self, num_classes: int = 2):
        # Create dummy model as ComposerClassifier needs params for optimizer
        net = torch.nn.Sequential(torch.nn.Linear(1, num_classes))
        super().__init__(module=net, num_classes=num_classes)

    def forward(self, x):
        out = torch.rand([x[1].shape[0], 2], dtype=x[0].dtype)
        out[:, 0] = 0.99
        out[:, 1] = 0.01
        return out


class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
        num_hidden (int): number of hidden units (default: 8)
        device (str): the device to initialize the model (default: 'cpu')
        bias (bool): whether or not to include bias in the linear layers (default: True)
    """

    def __init__(
        self,
        num_features: int = 1,
        num_classes: int = 2,
        num_hidden: int = 8,
        device: str = 'cpu',
        bias: bool = True,
    ) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, num_hidden, device=device, bias=bias)
        fc2 = torch.nn.Linear(num_hidden, num_classes, device=device, bias=bias)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        net.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]
        super().__init__(module=net, num_classes=num_classes)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = fc1
        self.fc2 = fc2

    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)


class SimpleMLP(torch.nn.Module):

    def __init__(self, num_features: int, device: str = 'cpu'):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc2 = torch.nn.Linear(num_features, num_features, device=device, bias=False)

        self.net = torch.nn.Sequential(self.fc1, torch.nn.ReLU(), self.fc2)

    def forward(self, x):
        return self.net(x)


# We use this Module to test state dict generation because fc1 and fc2
# are not submodules of EvenSimplerMLP, like they are in SimpleMLP.
class EvenSimplerMLP(torch.nn.Module):

    def __init__(self, num_features: int, device: str = 'cpu', num_out_features: int = 3):
        super().__init__()
        fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        fc2 = torch.nn.Linear(num_features, num_out_features, device=device, bias=False)

        self.module = torch.nn.Sequential(fc1, torch.nn.ReLU(), fc2)

    def forward(self, x):
        return self.module(x)


# This model is used when you want a SimpleMLP, but you want to explicitly
# test ComposerModels instead of nn.Module.
class SimpleComposerMLP(ComposerClassifier):

    def __init__(self, num_features: int, device: str, num_classes: int = 3):
        fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        fc2 = torch.nn.Linear(num_features, num_classes, device=device, bias=False)

        net = torch.nn.Sequential(fc1, torch.nn.ReLU(), fc2)
        super().__init__(num_classes=num_classes, module=net)


class SimpleWeightTiedModel(ComposerClassifier):
    """Small classification model with tied weights.
    Typically this model will be used to test weight tying w/ FSDP

    Args:
        num_features (int): number of input features (default: 1)
        tie_weights (bool): whether or not to tie weights (default: True)
        device (str): the device to initialize the model (default: 'cpu')
    """

    def __init__(self, num_features: int = 1, device: str = 'cpu') -> None:
        self.num_features = num_features

        mlp = SimpleMLP(num_features, device)

        net = torch.nn.Sequential(
            mlp,
            torch.nn.Softmax(dim=-1),
        )

        super().__init__(module=net, num_classes=num_features)

        self.mlp = mlp
        self.net = net
        self.net.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]

        self.mlp.fc1.weight = self.mlp.fc2.weight

    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)


class EmbeddedWeightTiedModel(ComposerClassifier):
    """A small classification model that consists of two simple MLPs,
    and we tie weights across the simple MLPs.
    Typically this model will be used to test weight tying w/ FSDP.

    Args:
        num_features (int): number of input features (default: 1)
        device (str): the device to initialize the model (default: 'cpu')
    """

    def __init__(self, num_features: int = 1, device: str = 'cpu') -> None:
        net1 = SimpleMLP(num_features, device)
        net2 = SimpleMLP(num_features, device)

        net = torch.nn.Sequential(
            net1,
            net2,
            torch.nn.Softmax(dim=-1),
        )

        super().__init__(module=net, num_classes=num_features)

        self.module.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]

        self.net1 = net1
        self.net2 = net2

        self.net1.fc1.weight = self.net2.fc1.weight

    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)


class SimpleConvModel(ComposerClassifier):
    """Small convolutional classifier.

    Args:
        num_channels (int): number of input channels (default: 3)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 2,
        norm: Optional[str] = None,
        norm_affine: bool = True,
    ) -> None:

        self.num_classes = num_classes
        self.num_channels = num_channels

        conv_args = {'kernel_size': (3, 3), 'padding': 1, 'stride': 2}
        conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=8, **conv_args)
        conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, **conv_args)
        norm_layer = None
        if norm is None:
            norm_layer = torch.nn.Identity()
        elif norm == 'batch':
            norm_layer = torch.nn.BatchNorm2d(4, affine=norm_affine)
        elif norm == 'instance':
            norm_layer = torch.nn.InstanceNorm2d(4, affine=norm_affine)
        elif norm == 'layer':
            norm_layer = torch.nn.LayerNorm(4, elementwise_affine=norm_affine)
        elif norm == 'group':
            norm_layer = torch.nn.GroupNorm(2, 4, affine=norm_affine)
        else:
            raise ValueError(f'Unknown norm: {norm}')
        pool = torch.nn.AdaptiveAvgPool2d(1)
        flatten = torch.nn.Flatten()
        fc1 = torch.nn.Linear(4, 16)
        fc2 = torch.nn.Linear(16, num_classes)

        net = torch.nn.Sequential(
            conv1,
            conv2,
            norm_layer,
            pool,
            flatten,
            fc1,
            fc2,
        )
        super().__init__(module=net, num_classes=self.num_classes)

        # bind these to class for access during surgery tests
        self.conv1 = conv1
        self.conv2 = conv2


class SimpleSegmentationModel(ComposerClassifier):
    """Small convolutional classifier.

    Args:
        num_channels (int): number of input channels (default: 3)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 2) -> None:
        self.num_classes = num_classes
        self.num_channels = num_channels

        conv_args = {'kernel_size': (3, 3), 'padding': 'same', 'stride': 1}
        conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=8, **conv_args)
        conv2 = torch.nn.Conv2d(in_channels=8, out_channels=num_classes, **conv_args)

        net = torch.nn.Sequential(
            conv1,
            conv2,
        )
        train_metrics = MetricCollection([CrossEntropy(), MIoU(num_classes)])
        val_metrics = MetricCollection([CrossEntropy(), MIoU(num_classes)])

        super().__init__(module=net, train_metrics=train_metrics, val_metrics=val_metrics)

        # bind these to class for access during surgery tests
        self.conv1 = conv1
        self.conv2 = conv2


class Mean(torch.nn.Module):

    def forward(self, x):
        return torch.mean(x, dim=1)


class SimpleTransformerBase(torch.nn.Module):
    """Base encoding transformer model for testing"""

    def __init__(self, vocab_size: int = 10, d_model: int = 16):
        super().__init__()
        embedding = torch.nn.Embedding(vocab_size, 16)
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=d_model, dropout=0.3)
        # necessary to make the model scriptable
        layer.__constants__ = []

        transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=torch.nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # necessary to make the model scriptable
        transformer.__constants__ = []

        self.net = torch.nn.Sequential(embedding, transformer)

        self.embedding = embedding
        self.transformer = transformer

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class SimpleTransformerMaskedLM(ComposerClassifier):

    def __init__(self, vocab_size: int = 10):
        self.vocab_size = vocab_size
        transformer_base = SimpleTransformerBase(vocab_size=vocab_size, d_model=16)
        lm_head = torch.nn.Linear(16, vocab_size)

        net = torch.nn.Sequential(transformer_base, lm_head)

        mlm_metrics = MetricCollection(LanguageCrossEntropy(ignore_index=-100), MaskedAccuracy(ignore_index=-100))
        loss = torch.nn.CrossEntropyLoss()
        super().__init__(module=net, train_metrics=mlm_metrics, val_metrics=mlm_metrics, loss_fn=loss)

        self.transformer_base = transformer_base
        self.lm_head = lm_head

    def loss(
        self,
        outputs: torch.Tensor,
        batch: Union[tuple[Any, torch.Tensor], dict[str, Any]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(batch, tuple):
            _, targets = batch
        else:
            targets = batch['labels']
        return self._loss_fn(outputs.view(-1, self.vocab_size), targets.view(-1), *args, **kwargs)

    def forward(self, batch: Union[tuple[torch.Tensor, Any], dict[str, Any]]) -> torch.Tensor:
        if isinstance(batch, tuple):
            inputs, _ = batch
        else:
            inputs = batch['input_ids']
        outputs = self.module(inputs)
        return outputs

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        if isinstance(batch, tuple):
            _, targets = batch
        else:
            targets = batch['labels']
        metric.update(outputs, targets)


class SimpleTransformerClassifier(ComposerClassifier):
    """Transformer model for testing"""

    def __init__(self, vocab_size: int = 10, num_classes: int = 2):
        transformer_base = SimpleTransformerBase(vocab_size=vocab_size, d_model=16)
        pooler = Mean()
        dropout = torch.nn.Dropout(0.3)
        classifier = torch.nn.Linear(16, num_classes)

        net = torch.nn.Sequential(transformer_base, pooler, dropout, classifier)

        super().__init__(module=net, num_classes=num_classes)

        self.transformer_base = transformer_base
        self.pooler = pooler
        self.classifier = classifier


class ConvModel(ComposerClassifier):
    """Convolutional network featuring strided convs, a batchnorm, max pooling, and average pooling."""

    def __init__(self) -> None:
        conv_args = {'kernel_size': (3, 3), 'padding': 1}
        conv1 = torch.nn.Conv2d(in_channels=32, out_channels=8, stride=2, bias=False, **conv_args)  # stride > 1
        conv2 = torch.nn.Conv2d(
            in_channels=8,
            out_channels=32,
            stride=2,
            bias=False,
            **conv_args,
        )  # stride > 1 but in_channels < 16
        conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, stride=1, bias=False, **conv_args)  # stride = 1
        bn = torch.nn.BatchNorm2d(num_features=64)
        pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        pool2 = torch.nn.AdaptiveAvgPool2d(1)
        flatten = torch.nn.Flatten()
        linear1 = torch.nn.Linear(64, 48)
        linear2 = torch.nn.Linear(48, 10)

        net = torch.nn.Sequential(
            conv1,
            conv2,
            conv3,
            bn,
            pool1,
            pool2,
            flatten,
            linear1,
            linear2,
        )

        super().__init__(module=net, num_classes=10)

        # bind these to class for access during surgery tests
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.bn = bn
        self.pool1 = pool1
        self.pool2 = pool2
        self.flatten = flatten
        self.linear1 = linear1
        self.linear2 = linear2


class SimpleModelWithDropout(ComposerClassifier):

    def __init__(self, num_features: int = 64, num_classes: int = 10) -> None:
        fc1 = torch.nn.Linear(num_features, 512)
        fc2 = torch.nn.Linear(512, num_classes)
        dropout = torch.nn.Dropout(0.5)

        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            dropout,
            fc2,
            torch.nn.Softmax(dim=-1),
        )

        super().__init__(module=net, num_classes=num_classes)

        self.fc1 = fc1
        self.fc2 = fc2
        self.dropout = dropout

    def loss(self, outputs: torch.Tensor, batch: tuple[Any, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        _, targets = batch
        targets = targets.squeeze(dim=0)
        return self._loss_fn(outputs, targets, *args, **kwargs)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        _, targets = batch
        metric.update(outputs.squeeze(dim=0), targets.squeeze(dim=0))

    def forward(self, batch: tuple[torch.Tensor, Any]) -> torch.Tensor:
        inputs, _ = batch
        inputs = inputs.squeeze(dim=0)
        outputs = self.module(inputs)
        return outputs


def composer_resnet(
    model_name: str,
    num_classes: int = 1000,
    weights: Optional[str] = None,
    groups: int = 1,
    width_per_group: int = 64,
    initializers: Optional[list[Initializer]] = None,
    loss_name: str = 'soft_cross_entropy',
) -> ComposerClassifier:
    """Helper function to create a :class:`.ComposerClassifier` with a torchvision ResNet model.
    From `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_ (He et al, 2015).
    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        weights (str, optional): If provided, pretrained weights can be specified, such as with ``IMAGENET1K_V2``. Default: ``None``.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (list[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.
        loss_name (str, optional): Loss function to use. E.g. 'soft_cross_entropy' or
            'binary_cross_entropy_with_logits'. Loss function must be in
            :mod:`~composer.loss.loss`. Default: ``'soft_cross_entropy'``".
    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with a torchvision ResNet model.
    """
    valid_model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if model_name not in valid_model_names:
        raise ValueError(f'model_name must be one of {valid_model_names} instead of {model_name}.')

    if loss_name not in loss_registry.keys():
        raise ValueError(
            f'Unrecognized loss function: {loss_name}. Please ensure the '
            'specified loss function is present in composer.loss.loss.py',
        )

    if initializers is None:
        initializers = []

    # Instantiate model
    model_fn = getattr(resnet, model_name)
    model = model_fn(weights=weights, num_classes=num_classes, groups=groups, width_per_group=width_per_group)

    # Grab loss function from loss registry
    loss_fn = loss_registry[loss_name]

    # Create metrics for train and validation
    train_metrics = MulticlassAccuracy(num_classes=num_classes, average='micro')
    val_metrics = MetricCollection([CrossEntropy(), MulticlassAccuracy(num_classes=num_classes, average='micro')])

    # Apply Initializers to model
    for initializer in initializers:
        initializer = Initializer(initializer)
        model.apply(initializer.get_initializer())

    composer_model = ComposerClassifier(model, train_metrics=train_metrics, val_metrics=val_metrics, loss_fn=loss_fn)
    return composer_model


# Note: These methods are an alternative to the tiny_bert fixtures in fixtures.py.
# Fixtures cannot be used natively as parametrized inputs, which we require when
# we wish to run a test across multiple models, one of which is a HuggingFace model.
# As a workaround, we inject objects into the PyTest namespace. Tests should not directly
# use pytest.{var}, but instead should import and use these helper copy methods so the
# objects in the PyTest namespace do not change.
def configure_tiny_bert_model() -> 'PreTrainedModel':
    try:
        from transformers import PreTrainedModel
        assert isinstance(pytest.tiny_bert_model, PreTrainedModel)
        return copy.deepcopy(pytest.tiny_bert_model)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_bert_tokenizer() -> Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast']:
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        assert isinstance(pytest.tiny_bert_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        return copy.deepcopy(pytest.tiny_bert_tokenizer)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_bert_config() -> 'PretrainedConfig':
    try:
        from transformers import PretrainedConfig
        assert isinstance(pytest.tiny_bert_config, PretrainedConfig)
        return copy.deepcopy(pytest.tiny_bert_config)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_bert_hf_model(use_logits: bool = True) -> HuggingFaceModel:
    return HuggingFaceModel(configure_tiny_bert_model(), configure_tiny_bert_tokenizer(), use_logits)


def configure_tiny_deberta_model() -> 'PreTrainedModel':
    try:
        from transformers import PreTrainedModel
        assert isinstance(pytest.tiny_deberta_model, PreTrainedModel)
        return copy.deepcopy(pytest.tiny_deberta_model)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_deberta_tokenizer() -> Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast']:
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        assert isinstance(pytest.tiny_deberta_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        return copy.deepcopy(pytest.tiny_deberta_tokenizer)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_deberta_config() -> 'PretrainedConfig':
    try:
        from transformers import PretrainedConfig
        assert isinstance(pytest.tiny_deberta_config, PretrainedConfig)
        return copy.deepcopy(pytest.tiny_deberta_config)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_deberta_hf_model(use_logits: bool = True) -> HuggingFaceModel:
    return HuggingFaceModel(
        configure_tiny_deberta_model(),
        configure_tiny_deberta_tokenizer(),
        use_logits,
    )


def configure_tiny_gpt2_model() -> 'PreTrainedModel':
    try:
        from transformers import PreTrainedModel
        assert isinstance(pytest.tiny_gpt2_model, PreTrainedModel)
        return copy.deepcopy(pytest.tiny_gpt2_model)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_gpt2_tokenizer() -> Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast']:
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        assert isinstance(pytest.tiny_gpt2_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        return copy.deepcopy(pytest.tiny_gpt2_tokenizer)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_gpt2_config() -> 'PretrainedConfig':
    try:
        from transformers import PretrainedConfig
        assert isinstance(pytest.tiny_gpt2_config, PretrainedConfig)
        return copy.deepcopy(pytest.tiny_gpt2_config)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_gpt2_hf_model(use_logits: bool = True) -> HuggingFaceModel:
    return HuggingFaceModel(configure_tiny_gpt2_model(), configure_tiny_gpt2_tokenizer(), use_logits)


def configure_tiny_t5_model() -> 'PreTrainedModel':
    try:
        from transformers import PreTrainedModel
        assert isinstance(pytest.tiny_t5_model, PreTrainedModel)
        return copy.deepcopy(pytest.tiny_t5_model)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_t5_tokenizer() -> Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast']:
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        assert isinstance(pytest.tiny_t5_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        return copy.deepcopy(pytest.tiny_t5_tokenizer)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_t5_config() -> 'PretrainedConfig':
    try:
        from transformers import PretrainedConfig
        assert isinstance(pytest.tiny_t5_config, PretrainedConfig)
        return copy.deepcopy(pytest.tiny_t5_config)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_t5_hf_model(use_logits: bool = True) -> HuggingFaceModel:
    return HuggingFaceModel(configure_tiny_t5_model(), configure_tiny_t5_tokenizer(), use_logits)


def configure_tiny_mpt_model() -> 'PreTrainedModel':
    try:
        from transformers import PreTrainedModel
        assert isinstance(pytest.tiny_mpt_model, PreTrainedModel)
        return copy.deepcopy(pytest.tiny_mpt_model)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_mpt_tokenizer() -> Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast']:
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        assert isinstance(pytest.tiny_mpt_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        return copy.deepcopy(pytest.tiny_mpt_tokenizer)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_mpt_config() -> 'PretrainedConfig':
    try:
        from transformers import PretrainedConfig
        assert isinstance(pytest.tiny_mpt_config, PretrainedConfig)
        return copy.deepcopy(pytest.tiny_mpt_config)
    except AttributeError:
        pytest.skip('Composer installed without NLP support')


def configure_tiny_mpt_hf_model(use_logits: bool = True) -> HuggingFaceModel:
    return HuggingFaceModel(configure_tiny_mpt_model(), configure_tiny_mpt_tokenizer(), use_logits)
