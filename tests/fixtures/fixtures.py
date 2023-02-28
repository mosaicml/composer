# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""These fixtures are shared globally across the test suite."""
import copy
import time

import coolname
import pytest
import torch
from torch.utils.data import DataLoader

from composer.core import State
from composer.devices import DeviceCPU, DeviceGPU
from composer.loggers import Logger
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.conftest import _get_option


@pytest.fixture
def rank_zero_seed(pytestconfig: pytest.Config) -> int:
    """Read the rank_zero_seed from the CLI option."""
    seed = _get_option(pytestconfig, 'seed', default='0')
    return int(seed)


@pytest.fixture
def minimal_state(rank_zero_seed: int, request: pytest.FixtureRequest):
    """Most minimally defined state possible.

    Tests should configure the state for their specific needs.
    """

    device = None
    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None

    return State(
        model=SimpleModel(),
        run_name='minimal_run_name',
        device=device,
        rank_zero_seed=rank_zero_seed,
        max_duration='100ep',
        dataloader=DataLoader(RandomClassificationDataset()),
        dataloader_label='train',
    )


@pytest.fixture()
def dummy_state(
    rank_zero_seed: int,
    request: pytest.FixtureRequest,
) -> State:

    model = SimpleModel()
    if request.node.get_closest_marker('gpu') is not None:
        # If using `dummy_state`, then not using the trainer, so move the model to the correct device
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    device = None
    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None

    state = State(
        model=model,
        run_name='dummy_run_name',
        device=device,
        precision='fp32',
        grad_accum=1,
        rank_zero_seed=rank_zero_seed,
        optimizers=optimizer,
        max_duration='10ep',
    )
    state.schedulers = scheduler
    state.set_dataloader(DataLoader(RandomClassificationDataset()), 'train')

    return state


@pytest.fixture
def empty_logger(minimal_state: State) -> Logger:
    """Logger without any output configured."""
    return Logger(state=minimal_state, destinations=[])


@pytest.fixture(scope='session')
def test_session_name(configure_dist: None) -> str:
    """Generate a random name for the test session that is the same on all ranks."""
    del configure_dist  # unused
    generated_session_name = str(int(time.time())) + '-' + coolname.generate_slug(2)
    name_list = [generated_session_name]
    # ensure all ranks have the same name
    dist.broadcast_object_list(name_list)
    return name_list[0]


@pytest.fixture
def sftp_uri():
    return 'sftp://localhost'


@pytest.fixture
def s3_bucket(request: pytest.FixtureRequest):
    if request.node.get_closest_marker('remote') is None:
        return 'my-bucket'
    else:
        return _get_option(request.config, 's3_bucket')


# Note: These session scoped fixtures should not be used directly in tests, but the non session scoped fixtures
# below should be used instead. This is because the session scoped fixtures return the same object to every
# test that requests it, so tests would have side effects on each other. Instead, the non session
# scoped fixtures below perform a deepcopy before returning the fixture.
def tiny_bert_model_helper(config):
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForMaskedLM.from_config(config)  # type: ignore (thirdparty)


@pytest.fixture(scope='session')
def _session_tiny_bert_model(_session_tiny_bert_config):  # type: ignore
    return tiny_bert_model_helper(_session_tiny_bert_config)


def tiny_bert_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    return transformers.AutoTokenizer.from_pretrained('bert-base-uncased')


@pytest.fixture(scope='session')
def _session_tiny_bert_tokenizer():  # type: ignore
    return tiny_bert_tokenizer_helper()


def tiny_bert_config_helper():
    transformers = pytest.importorskip('transformers')
    tiny_overrides = {
        'hidden_size': 128,
        'num_attention_heads': 2,
        'num_hidden_layers': 2,
        'intermediate_size': 512,
    }
    return transformers.AutoConfig.from_pretrained('bert-base-uncased', **tiny_overrides)


@pytest.fixture(scope='session')
def _session_tiny_bert_config():  # type: ignore
    return tiny_bert_config_helper()


def tiny_gpt2_model_helper(config):
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForCausalLM.from_config(config)


@pytest.fixture(scope='session')
def _session_tiny_gpt2_model(_session_tiny_gpt2_config):  # type: ignore
    return tiny_gpt2_model_helper(_session_tiny_gpt2_config)


def tiny_gpt2_config_helper():
    transformers = pytest.importorskip('transformers')

    tiny_overrides = {
        'n_embd': 2,
        'n_head': 2,
        'n_layer': 2,
    }
    return transformers.AutoConfig.from_pretrained('gpt2', **tiny_overrides)


@pytest.fixture(scope='session')
def _session_tiny_gpt2_config():  # type: ignore
    return tiny_gpt2_config_helper()


def tiny_gpt2_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return hf_tokenizer


@pytest.fixture(scope='session')
def _session_tiny_gpt2_tokenizer():  # type: ignore
    return tiny_gpt2_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_t5_config():  # type: ignore
    transformers = pytest.importorskip('transformers')

    tiny_overrides = {'d_ff': 128, 'd_model': 64, 'num_layers': 2, 'num_decoder_layers': 2, 'num_heads': 2}
    return transformers.AutoConfig.from_pretrained('t5-small', **tiny_overrides)


@pytest.fixture(scope='session')
def _session_tiny_t5_tokenizer():  # type: ignore
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512)
    return hf_tokenizer


@pytest.fixture(scope='session')
def _session_tiny_t5_model(_session_tiny_t5_config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.T5ForConditionalGeneration(config=_session_tiny_t5_config)


@pytest.fixture
def tiny_bert_model(_session_tiny_bert_model):
    return copy.deepcopy(_session_tiny_bert_model)


@pytest.fixture
def tiny_bert_tokenizer(_session_tiny_bert_tokenizer):
    return copy.deepcopy(_session_tiny_bert_tokenizer)


@pytest.fixture
def tiny_bert_config(_session_tiny_bert_config):
    return copy.deepcopy(_session_tiny_bert_config)


@pytest.fixture
def tiny_gpt2_config(_session_tiny_gpt2_config):
    return copy.deepcopy(_session_tiny_gpt2_config)


@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer):
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)


@pytest.fixture
def tiny_gpt2_model(_session_tiny_gpt2_model):
    return copy.deepcopy(_session_tiny_gpt2_model)


@pytest.fixture
def tiny_t5_config(_session_tiny_t5_config):
    return copy.deepcopy(_session_tiny_t5_config)


@pytest.fixture
def tiny_t5_tokenizer(_session_tiny_t5_tokenizer):
    return copy.deepcopy(_session_tiny_t5_tokenizer)


@pytest.fixture
def tiny_t5_model(_session_tiny_t5_model):
    return copy.deepcopy(_session_tiny_t5_model)
