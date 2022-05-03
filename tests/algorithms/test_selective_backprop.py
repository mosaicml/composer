# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Callable, Dict
from composer.datasets.dataloader import DataLoaderHparams

import pytest
import torch
from composer.algorithms import SelectiveBackpropHparams
from composer.algorithms.selective_backprop import SelectiveBackprop
from composer.algorithms.selective_backprop.selective_backprop import select_using_loss, should_selective_backprop, select_using_fn
from composer.core import Event
from composer.core.state import State
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.loggers import Logger
from composer.models import ComposerClassifier, BERTHparams, GPT2Hparams
from tests.utils import synthetic_utils
from tests.datasets import test_synthetic_lm_data
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder


@pytest.fixture
def N() -> int:
    """Batch size."""
    return 16


@pytest.fixture
def D() -> int:
    """Input dimension."""
    return 8


@pytest.fixture
def X(N: int, D: int) -> torch.Tensor:
    """2D input."""
    torch.manual_seed(42)
    return torch.randn(N, D)


@pytest.fixture
def X3D(N: int, D: int) -> torch.Tensor:
    """3D input."""
    torch.manual_seed(42)
    return torch.randn(N, D, D)


@pytest.fixture
def X5D(N: int, D: int) -> torch.Tensor:
    """5D input."""
    torch.manual_seed(42)
    return torch.randn(N, D, D, D, D)


@pytest.fixture
def Ximage(N: int) -> torch.Tensor:
    """4D image input."""
    torch.manual_seed(42)
    return torch.randn(N, 3, 32, 32)


@pytest.fixture
def y(N: int) -> torch.Tensor:
    """Target."""
    torch.manual_seed(42)
    return torch.randint(2, (N,))


@pytest.fixture
def loss_fun() -> Callable:
    """Fake loss function."""

    def loss(output, target, reduction="none"):
        return torch.ones_like(target)

    return loss


@pytest.fixture
def loss_fun_tuple() -> Callable:
    """Fake loss function that requires a batch tuple."""

    def loss(output, batch, reduction="none"):
        _, target = batch
        return torch.ones_like(target)

    return loss


@pytest.fixture
def bad_loss() -> Callable:
    """Fake loss function that will error."""

    def loss(output, target):
        return 0

    return loss


@pytest.fixture
def model(X: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model."""
    return torch.nn.Linear(X.shape[1], 1)


@pytest.fixture
def model3D(X3D: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model."""
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(X3D.shape[1], 1))


@pytest.fixture
def model5D(X5D: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model."""
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten(), torch.nn.Linear(X5D.shape[1], 1))


@pytest.fixture
def keep() -> float:
    """keep hparam."""
    return 0.5


@pytest.fixture
def scale_factor() -> float:
    """scale_factor hparam."""
    return 0.5


@pytest.fixture
def epoch() -> int:
    """Default epoch."""
    return 5


@pytest.fixture
def batch() -> int:
    """Default batch."""
    return 0

def make_dataset_configs():
    lm_dataset_configs = [config[0] for config in test_synthetic_lm_data.generate_parameter_configs( ['num_samples', 'chars_per_sample', 'column_names', 'tokenizer_family'])]
    for config in lm_dataset_configs:
        config['drop_last'] = False
        config['use_masked_lm'] = config['tokenizer_family'] == 'bert'
        if config['use_masked_lm']:
            config['mlm_probability'] = 0.15
    return lm_dataset_configs

def make_lm_tokenizer(config: Dict):
    dataset = synthetic_hf_dataset_builder(num_samples=config['num_samples'],
                                        chars_per_sample=config['chars_per_sample'],
                                        column_names=config['column_names'])
    tokenizer = generate_synthetic_tokenizer(config['tokenizer_family'], dataset)
    return tokenizer

def make_dummy_lm(model_name: str, max_position_embeddings, tokenizer):
    pytest.importorskip("transformers")
    if model_name == 'gpt2':
        class_name = GPT2Hparams
    elif model_name == 'bert':
        class_name = BERTHparams
    model_config = synthetic_utils.generate_dummy_model_config(class_name, tokenizer)
    if model_name == 'bert':
        model_config['num_labels'] = model_config['vocab_size']
        model_config['max_position_embeddings'] = max_position_embeddings
    model = class_name(model_config=model_config).initialize_object()
    return model

def synthetic_to_dataloader(dataset_config):
    """
    if tokenizer.pad_token_id is None:
        data_collator = transformers.default_data_collator
    else:
        print('using datacollecter for language modeling')
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                        mlm=dataset_config['use_masked_lm'],
                                                                        mlm_probability=dataset_config['mlm_probability'])
    sampler = dist.get_sampler(
            cast(Dataset, dataset),  # HF datasets do not subclass torch datasets, so this cast is needed
            drop_last=dataset_config['drop_last'],
            shuffle=True)
    """
    dataloader = LMDatasetHparams(use_synthetic=True, tokenizer_name=dataset_config['tokenizer_family'], use_masked_lm=dataset_config['use_masked_lm'], split='train', max_seq_length=dataset_config["chars_per_sample"])
    dataloader = dataloader.initialize_object(batch_size=dataset_config['num_samples'], dataloader_hparams=DataLoaderHparams())
    return dataloader

def minimal_lm_state(model, dataloader, rank_zero_seed=0):
    """Most minimally defined state possible.

    Tests should configure the state for their specific needs.
    """
    state = State(
        model=model,
        rank_zero_seed=rank_zero_seed,
        train_dataloader=dataloader,
        evaluators=[],
        max_duration='1ep',
    )
    state.batch = next(iter(state.train_dataloader)).data
    return state

@pytest.mark.parametrize("config", make_dataset_configs())
def test_minimal_lm_state(config):
    tokenizer = make_lm_tokenizer(config)
    lm = make_dummy_lm(config['tokenizer_family'], config['chars_per_sample'], tokenizer)
    dataloader = synthetic_to_dataloader(config)
    sample =  next(iter(dataloader)).data
    output = lm(sample)
    state = minimal_lm_state(lm, dataloader)
    assert hasattr(state, "batch")
    state_output = state.model(state.batch_dict)
    assert state_output.keys() == output.keys()
    assert state_output.loss.size() == output.loss.size()
    assert state_output.logits.size() == output.logits.size()
    assert state.batch_dict.keys() == sample.keys()
    for key in state.batch_dict.keys():
        assert state.batch_dict[key].size() == sample[key].size()


@pytest.fixture
def conv_model(Ximage: torch.Tensor, D: int) -> ComposerClassifier:
    """Dummy conv model."""
    return ComposerClassifier(torch.nn.Conv2d(Ximage.shape[1], D, 3))


@pytest.fixture
def state(minimal_state: State, conv_model: ComposerClassifier, loss_fun_tuple: Callable, epoch: int,
          batch: int) -> State:
    """State with required values set for Selective Backprop."""

    conv_model.loss = loss_fun_tuple
    minimal_state.model = conv_model

    minimal_state.timer.epoch._value = epoch
    minimal_state.timer.batch._value = epoch * minimal_state.steps_per_epoch + batch
    minimal_state.timer.batch_in_epoch._value = batch

    return minimal_state


def test_sb_hparams():
    hparams = SelectiveBackpropHparams(
        start=0.5,
        end=0.8,
        keep=0.5,
        scale_factor=0.5,
        interrupt=2,
    )
    algorithm = hparams.initialize_object()
    assert isinstance(algorithm, SelectiveBackprop)


# tests of the functional API
class TestSelectiveBackprop:

    @pytest.mark.parametrize("epoch,batch,interrupt", [(10, 0, 0), (10, 0, 2), (10, 2, 2)])
    def test_select_using_loss_true(self, epoch: int, batch: int, interrupt: int) -> None:
        """Test functional match when epoch is within interval."""
        start = 5
        end = 15
        is_chosen = should_selective_backprop(epoch, batch, start, end, interrupt)
        assert is_chosen

    @pytest.mark.parametrize("epoch,batch,interrupt", [(0, 0, 0), (20, 0, 0), (10, 1, 2)])
    def test_select_using_loss_false(self, epoch: int, batch: int, interrupt: int) -> None:
        """Test functional doesn't match when epoch is outside of interval."""
        start = 5
        end = 15
        is_chosen = should_selective_backprop(epoch, batch, start, end, interrupt)
        assert not is_chosen

    @pytest.mark.parametrize("keep", [0.5])
    @pytest.mark.parametrize("scale_factor", [0.5])
    @pytest.mark.xfail()
    def test_selective_output_shape_3D(self, X3D: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                       loss_fun: Callable, keep: float, scale_factor: float) -> None:
        """Test functional selection on 3D inputs."""
        N, D, _ = X3D.shape

        X_scaled, y_scaled = select_using_loss(X3D, y, model, loss_fun, keep, scale_factor)
        assert X_scaled.shape == (int(N * keep), D, D)
        assert y_scaled.shape == (int(N * keep),)

    @pytest.mark.parametrize("keep", [1, 0.5, 0.75])
    @pytest.mark.parametrize("scale_factor", [1])
    def test_selective_output_shape(self, X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, loss_fun: Callable,
                                    keep: float, scale_factor: float) -> None:
        """Test functional selection on 2D inputs."""
        N, D = X.shape

        X_scaled, y_scaled = select_using_loss(X, y, model, loss_fun, keep, scale_factor)
        assert X_scaled.shape == (int(N * keep), D)
        assert y_scaled.shape == (int(N * keep),)

    @pytest.mark.parametrize("keep", [0.5, 0.75, 1])
    @pytest.mark.parametrize("scale_factor", [0.5, 0.75])
    def test_selective_output_shape_scaled(self, Ximage: torch.Tensor, y: torch.Tensor, conv_model: ComposerClassifier,
                                           loss_fun: Callable, keep: float, scale_factor: float) -> None:
        """Test functional selection on 4D inputs."""
        N, C, H, W = Ximage.shape
        X_scaled, y_scaled = select_using_loss(Ximage, y, conv_model.module, loss_fun, keep, scale_factor)
        assert X_scaled.shape == (int(N * keep), C, H, W)
        assert y_scaled.shape == (int(N * keep),)

    def test_selective_backprop_interp_dim_error(self, X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                                 loss_fun: Callable) -> None:
        """Ensure that ValueError is raised when input tensor can't be scaled."""
        with pytest.raises(ValueError):
            select_using_loss(X, y, model, loss_fun, 1, 0.5)

    def test_selective_backprop_bad_loss_error(self, X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                               bad_loss: Callable) -> None:
        """Ensure that ValueError is raised when loss function doesn't have `reduction` kwarg."""
        with pytest.raises(TypeError) as execinfo:
            select_using_loss(X, y, model, bad_loss, 1, 1)
        MATCH = "must take a keyword argument `reduction`."
        assert MATCH in str(execinfo.value)


"""
Test Selective Backprop Algorithm
"""


class TestSelectiveBackpropAlgorithm:

    @pytest.fixture
    def sb_algorithm(self, scale_factor, keep) -> SelectiveBackprop:
        return SelectiveBackprop(
            start=0.5,
            end=0.8,
            keep=keep,
            scale_factor=scale_factor,
            interrupt=2,
        )

    @pytest.mark.parametrize("event", [Event.AFTER_DATALOADER])
    @pytest.mark.parametrize("epoch,batch", [(5, 0), (7, 0), (5, 2)])
    def test_match_correct(self, event: Event, sb_algorithm: SelectiveBackprop, state: State) -> None:
        """Algo should match AFTER_DATALOADER in the right interval."""
        state.max_duration = "10ep"

        assert sb_algorithm.match(event, state)

    @pytest.mark.parametrize("event,epoch,batch", [(Event.AFTER_DATALOADER, 0, 0), (Event.AFTER_DATALOADER, 5, 1)])
    def test_match_incorrect(self, event: Event, sb_algorithm: SelectiveBackprop, state: State) -> None:
        """Algo should NOT match the wrong interval."""
        state.max_duration = "10ep"

        assert not sb_algorithm.match(event, state)

    @pytest.mark.parametrize("epoch,batch", [(5, 0)])
    @pytest.mark.parametrize("keep", [0.5, 0.75, 1])
    @pytest.mark.parametrize("scale_factor", [0.5, 1])
    def test_apply(self, Ximage: torch.Tensor, y: torch.Tensor, sb_algorithm: SelectiveBackprop, state: State,
                   empty_logger: Logger, keep: float) -> None:
        """Test apply with image inputs gives the right output shape."""
        N, C, H, W = Ximage.shape

        state.max_duration = "10ep"
        state.batch = (Ximage, y)
        sb_algorithm.apply(Event.INIT, state, empty_logger)
        sb_algorithm.apply(Event.AFTER_DATALOADER, state, empty_logger)

        X_scaled, y_scaled = state.batch
        assert X_scaled.shape == (int(N * keep), C, H, W)
        assert y_scaled.shape == (int(N * keep),)
