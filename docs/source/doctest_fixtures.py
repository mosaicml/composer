# Copyright 2021 MosaicML. All Rights Reserved.

"""
Fixtures available in doctests.

The script is run before any doctests are executed,
so all imports and variables are available in any doctest.
The output of this setup script does not show up in the documentation.
"""
import os
import sys
import tempfile
from typing import Any
from typing import Callable as Callable

import numpy as np
import torch.optim
import torch.utils.data
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

import composer
import composer.trainer
import composer.trainer.trainer
from composer import Trainer as OriginalTrainer
from composer.core import Algorithm as Algorithm
from composer.core import Callback as Callback
from composer.core import DataSpec as DataSpec
from composer.core import Engine as Engine
from composer.core import Evaluator as Evaluator
from composer.core import Event as Event
from composer.core import State as State
from composer.core import Time as Time
from composer.core import Timer as Timer
from composer.core import Timestamp as Timestamp
from composer.core import TimeUnit as TimeUnit
from composer.core import types as types
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.loggers import InMemoryLogger as InMemoryLogger
from composer.loggers import Logger as Logger
from composer.loggers import LogLevel as LogLevel
from composer.models import ComposerModel as ComposerModel
from composer.optim.scheduler import ConstantScheduler
from composer.utils import ensure_tuple as ensure_tuple

# Need to insert the repo root at the beginning of the path, since there may be other modules named `tests`
# Assuming that docs generation is running from the `docs` directory
_docs_dir = os.path.abspath(".")
_repo_root = os.path.dirname(_docs_dir)
if sys.path[0] != _repo_root:
    sys.path.insert(0, _repo_root)

from tests.fixtures.models import SimpleBatchPairModel

# Change the cwd to be the tempfile, so we don't pollute the documentation source folder
tmpdir = tempfile.TemporaryDirectory()
cwd = os.path.abspath(".")
os.chdir(tmpdir.name)

num_channels = 3
num_classes = 10
data_shape = (num_channels, 5, 5)

model = SimpleBatchPairModel(num_channels, num_classes)

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
    optimizers=optimizer,
    grad_accum=1,
    train_dataloader=train_dataloader,
    evaluators=[],
    max_duration="1ep",
    precision="fp32",
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


# patch the Trainer to accept ellipses and bind the required arguments to the Trainer
# so it can be used without arguments in the doctests
def Trainer(fake_ellipses: None = None, **kwargs: Any):
    del fake_ellipses  # unused
    if "model" not in kwargs:
        kwargs["model"] = model
    if "optimizers" not in kwargs:
        kwargs["optimizers"] = torch.optim.SGD(kwargs["model"].parameters(), lr=0.01)
    if "schedulers" not in kwargs:
        kwargs["schedulers"] = ConstantScheduler()
    if "max_duration" not in kwargs:
        kwargs["max_duration"] = "1ep"
    if "train_dataloader" not in kwargs:
        kwargs["train_dataloader"] = train_dataloader
    if "eval_dataloader" not in kwargs:
        kwargs["eval_dataloader"] = eval_dataloader
    if "loggers" not in kwargs:
        kwargs["loggers"] = []  # hide tqdm logging
    return OriginalTrainer(**kwargs)


# patch composer so that 'from composer import Trainer' calls do not override change above
composer.Trainer = Trainer
composer.trainer.Trainer = Trainer
composer.trainer.trainer.Trainer = Trainer
