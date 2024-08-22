# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.remote
def test_azure_object_store_integration():
    model = SimpleModel()
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    trainer_save = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        save_folder='azure://mosaicml-composer-tests/checkpoints/{run_name}',
        save_filename='test-model.pt',
        max_duration='1ba',
    )
    run_name = trainer_save.state.run_name
    trainer_save.fit()
    trainer_save.close()

    trainer_load = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        load_path=f'azure://mosaicml-composer-tests/checkpoints/{run_name}/test-model.pt',
        max_duration='2ba',
    )
    trainer_load.fit()
    trainer_load.close()
