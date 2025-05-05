# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""These fixtures are shared globally across the test suite."""
import copy
import hashlib
import os
import time
import zipfile

import coolname
import pytest
import requests
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
        device_train_microbatch_size=1,
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
        return os.environ.get('S3_BUCKET', 'mosaicml-internal-integration-testing')


@pytest.fixture
def s3_ephemeral_prefix():
    '''Objects under this prefix purged according to the bucket's lifecycle policy.'''
    return 'ephemeral'


@pytest.fixture
def s3_read_only_prefix():
    '''Tests can only read from this prefix, but it won't ever be purged.'''
    return 'read_only'


## MODEL HELPERS ##
def causal_lm_model_helper(config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForCausalLM.from_config(config)


def masked_lm_model_helper(config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForMaskedLM.from_config(
        config,
    )  # type: ignore (thirdparty)


def tiny_t5_model_helper(config):
    transformers = pytest.importorskip('transformers')

    return transformers.T5ForConditionalGeneration(config=config)  # type: ignore (thirdparty)


## CONFIG HELPERS ##
def tiny_gpt2_config_helper():
    pytest.importorskip('transformers')
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    config_dict = {
        'activation_function': 'gelu_new',
        'architectures': ['GPT2LMHeadModel',],
        'attn_pdrop': 0.1,
        'bos_token_id': 50256,
        'embd_pdrop': 0.1,
        'eos_token_id': 50256,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-05,
        'model_type': 'gpt2',
        'n_ctx': 1024,
        'n_embd': 2,
        'n_head': 2,
        'n_layer': 2,
        'n_positions': 1024,
        'resid_pdrop': 0.1,
        'summary_activation': None,
        'summary_first_dropout': 0.1,
        'summary_proj_to_labels': True,
        'summary_type': 'cls_index',
        'summary_use_proj': True,
        'task_specific_params': {
            'text-generation': {
                'do_sample': True,
                'max_length': 50,
            },
        },
        'vocab_size': 50258,
    }

    config_object = GPT2Config(
        **config_dict,
    )
    return config_object


def tiny_codellama_config_helper(tie_word_embeddings: bool = False):
    pytest.importorskip('transformers')
    from transformers.models.llama.configuration_llama import LlamaConfig

    config_dict = {
        '_name_or_path': 'codellama/CodeLlama-7b-hf',
        'architectures': ['LlamaForCausalLM',],
        'bos_token_id': 1,
        'eos_token_id': 2,
        'hidden_act': 'silu',
        'hidden_size': 32,
        'initializer_range': 0.02,
        'intermediate_size': 64,
        'max_position_embeddings': 16384,
        'model_type': 'llama',
        'num_attention_heads': 32,
        'num_hidden_layers': 2,
        'num_key_value_heads': 32,
        'pretraining_tp': 1,
        'rms_norm_eps': 1e-05,
        'rope_scaling': None,
        'rope_theta': 1000000,
        'tie_word_embeddings': tie_word_embeddings,
        'torch_dtype': 'bfloat16',
        'transformers_version': '4.33.0.dev0',
        'use_cache': True,
        'vocab_size': 32016,
    }

    config_object = LlamaConfig(
        **config_dict,
    )
    return config_object


def tiny_bert_config_helper():
    pytest.importorskip('transformers')
    from transformers.models.bert.configuration_bert import BertConfig

    config_object = {
        'architectures': ['BertForMaskedLM',],
        'attn_implementation': 'eager',
        'attention_probs_dropout_prob': 0.1,
        'gradient_checkpointing': False,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'hidden_size': 128,
        'initializer_range': 0.02,
        'intermediate_size': 512,
        'layer_norm_eps': 1e-12,
        'max_position_embeddings': 512,
        'model_type': 'bert',
        'num_attention_heads': 2,
        'num_hidden_layers': 2,
        'pad_token_id': 0,
        'position_embedding_type': 'absolute',
        'transformers_version': '4.6.0.dev0',
        'type_vocab_size': 2,
        'use_cache': True,
        'vocab_size': 30522,
    }

    config_object = BertConfig(
        **config_object,
    )
    return config_object


def tiny_t5_config_helper():
    pytest.importorskip('transformers')
    from transformers.models.t5.configuration_t5 import T5Config

    config_object = {
        'architectures': ['T5ForConditionalGeneration',],
        'd_ff': 128,
        'd_kv': 64,
        'd_model': 64,
        'decoder_start_token_id': 0,
        'dropout_rate': 0.1,
        'eos_token_id': 1,
        'initializer_factor': 1.0,
        'is_encoder_decoder': True,
        'layer_norm_epsilon': 1e-06,
        'model_type': 't5',
        'n_positions': 512,
        'num_heads': 2,
        'num_layers': 2,
        'num_decoder_layers': 2,
        'output_past': True,
        'pad_token_id': 0,
        'relative_attention_num_buckets': 32,
        'task_specific_params': {
            'summarization': {
                'early_stopping': True,
                'length_penalty': 2.0,
                'max_length': 200,
                'min_length': 30,
                'no_repeat_ngram_size': 3,
                'num_beams': 4,
                'prefix': 'summarize: ',
            },
            'translation_en_to_de': {
                'early_stopping': True,
                'max_length': 300,
                'num_beams': 4,
                'prefix': 'translate English to German: ',
            },
            'translation_en_to_fr': {
                'early_stopping': True,
                'max_length': 300,
                'num_beams': 4,
                'prefix': 'translate English to French: ',
            },
            'translation_en_to_ro': {
                'early_stopping': True,
                'max_length': 300,
                'num_beams': 4,
                'prefix': 'translate English to Romanian: ',
            },
        },
        'vocab_size': 32128,
    }

    return T5Config(**config_object)


def assets_path():
    rank = os.environ.get('RANK', '0')
    folder_name = 'tokenizers' + (f'_{rank}' if rank != '0' else '')
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', folder_name)


@pytest.fixture(scope='session')
def tokenizers_assets():
    download_tokenizers_files()


def download_tokenizers_files():
    """Download the tokenizers assets

    We download from github, because downloading from HF directly is flaky and gets rate limited easily.

    Raises:
        ValueError: If the checksum of the downloaded file does not match the expected checksum.
    """
    # Define paths
    tokenizers_dir = assets_path()

    if os.path.exists(tokenizers_dir):
        return

    # Create assets directory if it doesn't exist
    os.makedirs(tokenizers_dir, exist_ok=True)

    # URL for the tokenizers.zip file
    url = 'https://github.com/mosaicml/ci-testing/releases/download/tokenizers/tokenizers.zip'
    expected_checksum = '12dc1f254270582f7806588f1f1d47945590c5b42dee28925e5dab95f2d08075'

    # Download the zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    zip_path = os.path.join(tokenizers_dir, 'tokenizers.zip')

    # Check the checksum
    checksum = hashlib.sha256(response.content).hexdigest()
    if checksum != expected_checksum:
        raise ValueError(f'Checksum mismatch: expected {expected_checksum}, got {checksum}')

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file
    print(f'Extracting tokenizers.zip to {tokenizers_dir}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tokenizers_dir)

    # Optionally remove the zip file after extraction
    os.remove(zip_path)


## TOKENIZER HELPERS ##
def assets_tokenizer_helper(name: str):
    """Load a tokenizer from the assets directory."""
    transformers = pytest.importorskip('transformers')

    download_tokenizers_files()

    assets_dir = assets_path()
    tokenizer_path = os.path.join(assets_dir, name)

    # Load the tokenizer
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    return hf_tokenizer


## SESSION MODELS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_model(_session_tiny_gpt2_config):  # type: ignore
    return causal_lm_model_helper(_session_tiny_gpt2_config)


@pytest.fixture(scope='session')
def _session_tiny_bert_model(_session_tiny_bert_config):  # type: ignore
    return masked_lm_model_helper(_session_tiny_bert_config)


@pytest.fixture(scope='session')
def _session_tiny_t5_model(_session_tiny_t5_config):  # type: ignore
    return tiny_t5_model_helper(_session_tiny_t5_config)


## SESSION CONFIGS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_config():  # type: ignore
    return tiny_gpt2_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_bert_config():  # type: ignore
    return tiny_bert_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_t5_config():  # type: ignore
    return tiny_t5_config_helper()


## SESSION TOKENIZERS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_tokenizer(tokenizers_assets):  # type: ignore
    tokenizer = assets_tokenizer_helper('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


@pytest.fixture(scope='session')
def _session_tiny_t5_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('t5')


@pytest.fixture(scope='session')
def _session_tiny_bert_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('bertt')


@pytest.fixture(scope='session')
def _session_tiny_t0_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('t0pp')


## MODEL FIXTURES ##
@pytest.fixture
def tiny_bert_model(_session_tiny_bert_model):  # type: ignore
    return copy.deepcopy(_session_tiny_bert_model)


@pytest.fixture
def tiny_gpt2_model(_session_tiny_gpt2_model):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_model)


@pytest.fixture
def tiny_t5_model(_session_tiny_t5_model):  # type: ignore
    return copy.deepcopy(_session_tiny_t5_model)


## CONFIG FIXTURES ##
@pytest.fixture
def tiny_bert_config(_session_tiny_bert_config):  # type: ignore
    return copy.deepcopy(_session_tiny_bert_config)


def _gpt2_peft_config():
    pytest.importorskip('peft')
    from peft import get_peft_config

    peft_config = get_peft_config({
        'peft_type': 'LORA',
        'task_type': 'CAUSAL_LM',
        'target_modules': ['c_attn'],
        'fan_in_fan_out': True,
    })
    return peft_config


@pytest.fixture
def gpt2_peft_config():
    return _gpt2_peft_config()


## TOKENIZER FIXTURES ##
@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)


@pytest.fixture
def tiny_t5_tokenizer(_session_tiny_t5_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_t5_tokenizer)


@pytest.fixture
def tiny_bert_tokenizer(_session_tiny_bert_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_bert_tokenizer)


@pytest.fixture
def tiny_mpt_tokenizer(_session_tiny_mpt_tokenizer):
    return copy.deepcopy(_session_tiny_mpt_tokenizer)


@pytest.fixture
def tiny_t0_tokenizer(_session_tiny_t0_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_t0_tokenizer)


@pytest.fixture
def clean_mlflow_runs():
    """Clean up MLflow runs before and after tests.

    This fixture ensures no MLflow runs persist between tests,
    which prevents "Run already active" errors.
    """
    try:
        import mlflow
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass

        yield

        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
    except ImportError:
        yield
