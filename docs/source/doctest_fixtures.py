# Copyright 2021 MosaicML. All Rights Reserved.

"""
Fixtures available in doctests.

The script is run before any doctests are executed,
so all imports and variables are available in any doctest.
The output of this setup script does not show up in the documentation.
"""
import functools
import os
import sys

import torch.optim
import torch.utils.data

from composer import *  # Make all composer imports available in doctests
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.trainer import Trainer
from composer.utils import *  # Make all composer.utils imports available in doctests

# Need to insert the repo root at the beginning of the path, since there may be other modules named `tests`
# Assuming that docs generation is running from the `docs` directory
_docs_dir = os.path.abspath(".")
_repo_root = os.path.dirname(_docs_dir)
if sys.path[0] != _repo_root:
    sys.path.insert(0, _repo_root)

from tests.fixtures.models import SimpleBatchPairModel

num_channels = 3
num_classes = 10
data_shape = (num_channels, 5, 5)

model = SimpleBatchPairModel(num_channels, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

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

# bind the required arguments to the Trainer so it can be used without arguments in the doctests
Trainer = functools.partial(
    Trainer,
    model=model,
    max_duration="1ep",
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
)
