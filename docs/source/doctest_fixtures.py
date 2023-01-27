# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# disabling general type issues because of monkeypatching
#yright: reportGeneralTypeIssues=none

"""Fixtures available in doctests.

The script is run before any doctests are executed,
so all imports and variables are available in any doctest.
The output of this setup script does not show up in the documentation.
"""
import logging

logging.basicConfig(level=logging.WARN)
import os
import sys
import tempfile
from typing import Any
from typing import Callable as Callable
from urllib.parse import urlparse

import numpy as np
import pytest
import torch
import torch.optim
import torch.utils.data
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

import composer
import composer.loggers
import composer.loggers.remote_uploader_downloader
import composer.trainer
import composer.trainer.trainer
import composer.utils
import composer.utils.checkpoint
import composer.utils.file_helpers
from composer import Trainer
from composer.core import Algorithm as Algorithm
from composer.core import Callback as Callback
from composer.core import DataSpec as DataSpec
from composer.core import Engine as Engine
from composer.core import Evaluator as Evaluator
from composer.core import Event as Event
from composer.core import State as State
from composer.core import Time as Time
from composer.core import Timestamp as Timestamp
from composer.core import TimeUnit as TimeUnit
from composer.core import types as types
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.devices import DeviceCPU
from composer.loggers import InMemoryLogger as InMemoryLogger
from composer.loggers import Logger as Logger
from composer.loggers import RemoteUploaderDownloader
from composer.models import ComposerModel as ComposerModel
from composer.optim.scheduler import ConstantScheduler
from composer.utils import LibcloudObjectStore
from composer.utils import ensure_tuple as ensure_tuple

try:
    import wandb
    _WANDB_INSTALLED = True
    del wandb  # unused
except ImportError:
    _WANDB_INSTALLED = False

try:
    import comet_ml
    _COMETML_INSTALLED = True
    del comet_ml  # unused
except ImportError:
    _COMETML_INSTALLED = False

try:
    import libcloud
    _LIBCLOUD_INSTALLED = True
    del libcloud  # unused
except ImportError:
    _LIBCLOUD_INSTALLED = False

# Need to insert the repo root at the beginning of the path, since there may be other modules named `tests`
# Assuming that docs generation is running from the `docs` directory
_docs_dir = os.path.abspath('.')
_repo_root = os.path.dirname(_docs_dir)
if sys.path[0] != _repo_root:
    sys.path.insert(0, _repo_root)

from tests.common import SimpleModel
from tests.common.datasets import RandomTextClassificationDataset

# Disable wandb
os.environ['WANDB_MODE'] = 'disabled'

# Change the cwd to be the tempfile, so we don't pollute the documentation source folder
tmpdir = tempfile.mkdtemp()
cwd = os.path.abspath('.')
os.chdir(tmpdir)

num_channels = 3
num_classes = 10
data_shape = (num_channels, 5, 5)

Model = SimpleModel

model = SimpleModel(num_channels, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

scheduler = CosineAnnealingLR(optimizer, T_max=1)

dataset = SyntheticBatchPairDataset(
    total_dataset_size=100,
    data_shape=data_shape,
    num_classes=num_classes,
    num_unique_samples_to_create=10,
)

train_dataset = dataset
eval_dataset = dataset

batch_size = 10

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)

eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)

state = State(
    rank_zero_seed=0,
    model=model,
    run_name='run_name',
    device=DeviceCPU(),
    optimizers=optimizer,
    grad_accum=1,
    dataloader=train_dataloader,
    dataloader_label='train',
    max_duration='1ep',
    precision='fp32',
)

logger = Logger(state)

engine = Engine(state, logger)

image = Image.fromarray(np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8))

# error: "randn" is not a known member of module (reportGeneralTypeIssues)
X_example = torch.randn(batch_size, num_channels, 32, 32)  # type: ignore
# error: "randn" is not a known member of module (reportGeneralTypeIssues)
logits = torch.randn(batch_size, num_classes)  # type: ignore
# error: "randint" is not a known member of module (reportGeneralTypeIssues)
y_example = torch.randint(num_classes, (batch_size,))  # type: ignore


def loss_fun(output, target, reduction='none'):
    """Dummy loss function."""
    return torch.ones_like(target)


# Patch Trainer __init__ function to replace arguments while preserving type
_original_trainer_init = Trainer.__init__


def _new_trainer_init(self, fake_ellipses: None = None, **kwargs: Any):
    if 'model' not in kwargs:
        kwargs['model'] = model
    if 'optimizers' not in kwargs:
        kwargs['optimizers'] = torch.optim.SGD(kwargs['model'].parameters(), lr=0.01)
    if 'schedulers' not in kwargs:
        kwargs['schedulers'] = ConstantScheduler()
    if 'max_duration' not in kwargs:
        kwargs['max_duration'] = '1ep'
    if 'train_dataloader' not in kwargs:
        kwargs['train_dataloader'] = train_dataloader
    if 'eval_dataloader' not in kwargs:
        kwargs['eval_dataloader'] = eval_dataloader
    if 'progress_bar' not in kwargs:
        kwargs['progress_bar'] = False  # hide tqdm logging
    if 'log_to_console' not in kwargs:
        kwargs['log_to_console'] = False  # hide console logging
    if 'save_folder' in kwargs and urlparse(kwargs['save_folder']).scheme == 'gs':
        os.environ['GCS_KEY'] = 'foo'
        os.environ['GCS_SECRET'] = 'foo'
    if 'load_path' in kwargs and urlparse(kwargs['load_path']).scheme in ['s3', 'oci', 'gs']:
        if urlparse(kwargs['load_path']).scheme == 'gs':
            os.environ['GCS_KEY'] = 'foo'
            os.environ['GCS_SECRET'] = 'foo'
        kwargs['load_path'] = urlparse(kwargs['load_path']).path.lstrip('/')
        kwargs['load_object_store'] = LibcloudObjectStore()
    _original_trainer_init(self, **kwargs)


Trainer.__init__ = _new_trainer_init


# Do not attempt to validate cloud credentials
def _do_not_validate(*args, **kwargs) -> None:
    pass


composer.loggers.remote_uploader_downloader._validate_credentials = _do_not_validate  # type: ignore

# Patch RemoteUploaderDownloader __init__ function to replace arguments while preserving type
_original_RemoteUploaderDownloader_init = RemoteUploaderDownloader.__init__


def _new_RemoteUploaderDownloader_init(self, fake_ellipses: None = None, **kwargs: Any):
    os.makedirs('./object_store', exist_ok=True)
    kwargs.update(use_procs=False,
                  num_concurrent_uploads=1,
                  bucket_uri='libcloud://.',
                  backend_kwargs={
                      'provider': 'local',
                      'container': '.',
                      'provider_kwargs': {
                          'key': os.path.abspath('./object_store'),
                      },
                  })
    _original_RemoteUploaderDownloader_init(self, **kwargs)


RemoteUploaderDownloader.__init__ = _new_RemoteUploaderDownloader_init  # type: ignore

# Patch ObjectStore __init__ function to replace arguments while preserving type
_original_libcloudObjectStore_init = LibcloudObjectStore.__init__


def _new_libcloudObjectStore_init(self, fake_ellipses: None = None, **kwargs: Any):
    os.makedirs('./object_store', exist_ok=True)
    kwargs.update(
        provider='local',
        container='.',
        provider_kwargs={
            'key': os.path.abspath('./object_store'),
        },
    )
    _original_libcloudObjectStore_init(self, **kwargs)


LibcloudObjectStore.__init__ = _new_libcloudObjectStore_init  # type: ignore

# Note: These methods are an alternative to the tiny_bert fixtures in fixtures.py.
# Fixtures cannot be used natively as parametrized inputs, which we require when
# we wish to run a test across multiple models, one of which is a HuggingFace BERT Tiny.
# As a workaround, we inject objects into the PyTest namespace. Tests should not directly
# use pytest.{var}, but instead should import and use the helper copy methods configure_{var}
# (in tests.common.models) so the objects in the PyTest namespace do not change.
try:
    import transformers
    del transformers
    TRANSFORMERS_INSTALLED = True
except ImportError:
    TRANSFORMERS_INSTALLED = False

if TRANSFORMERS_INSTALLED:
    from tests.fixtures.fixtures import (tiny_bert_config_helper, tiny_bert_model_helper, tiny_bert_tokenizer_helper,
                                         tiny_gpt2_config_helper, tiny_gpt2_model_helper, tiny_gpt2_tokenizer_helper)
    pytest.tiny_bert_config = tiny_bert_config_helper()  # type: ignore
    pytest.tiny_bert_model = tiny_bert_model_helper(pytest.tiny_bert_config)  # type: ignore
    pytest.tiny_bert_tokenizer = tiny_bert_tokenizer_helper()  # type: ignore
    pytest.tiny_gpt2_config = tiny_gpt2_config_helper()  # type: ignore
    pytest.tiny_gpt2_model = tiny_gpt2_model_helper(pytest.tiny_gpt2_config)  # type: ignore
    pytest.tiny_gpt2_tokenizer = tiny_gpt2_tokenizer_helper()  # type: ignore
