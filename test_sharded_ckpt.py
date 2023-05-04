import pathlib
from typing import Sequence

import pytest
import torch
import torch.distributed
from packaging import version
from torch.utils.data import DataLoader, Dataset

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.markers import world_size


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.
    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        return self.x[index], self.y[index]


class SimpleModel(ComposerClassifier):
    """Small classification model.
    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 32, num_hidden=16, num_classes: int = 8) -> None:

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(num_features, num_hidden, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_classes, bias=False),
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net, num_classes=num_classes)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        # self.fc1 = fc1
        # self.fc2 = fc2


def get_trainer(
    dataset,
    dataloader,
    save_folder=None,
    save_filename='ba{batch}-rank{rank}.pt',
    num_features=32,
    num_hidden=16,
    num_classes=8,
    fsdp_state_dict_type='full',
    load_path=None,
    autoresume=False,
    run_name=None,
    python_log_level=None,
    max_duration='2ba',
    save_num_checkpoints_to_keep=-1,
    save_weights_only=False,
    load_weights_only=False,
    log_to_console=False,
    save_interval='2ba',
):
    model = SimpleModel(num_features=num_features, num_hidden=num_hidden, num_classes=num_classes)
    optim = torch.optim.Adam(params=model.parameters())
    trainer = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={
            'min_params': 1,
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': 'FULL_SHARD'
        },
        save_folder=save_folder,
        max_duration=max_duration,
        save_filename=save_filename,
        save_overwrite=False,
        save_weights_only=save_weights_only,
        load_path=load_path,
        load_weights_only=load_weights_only,
        progress_bar=False,
        log_to_console=log_to_console,
        autoresume=autoresume,
        run_name=run_name,
        save_interval=save_interval,
        python_log_level=python_log_level,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
    )
    return trainer


if __name__ == '__main__':

    import time

    # from torch.distributed import checkpoint as dist_cp
    # from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    # from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    # from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    # from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    from composer.core.state import fsdp_state_dict_type_context
    num_features = 16
    num_classes = 8
    dataset = RandomClassificationDataset(shape=(num_features, 1, 1), num_classes=num_classes, size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=16)
    ## Save
    s3_folder = 's3://mosaicml-internal-checkpoints-test/evan-test/test_sharded_checkpoints/{run_name}'
    local_folder = 'test_checkpoints/{run_name}'
    local_copy_of_s3_folder = './evan-test/test_sharded_checkpoints/{run_name}'
    trainer = get_trainer(dataset,
                          dataloader,
                          num_features=num_features,
                          num_classes=num_classes,
                          save_folder=local_folder,
                          autoresume=False,
                          run_name='ar-testy-test',
                          save_weights_only=False,
                          max_duration='4ba',
                          fsdp_state_dict_type='sharded',
                          save_num_checkpoints_to_keep=-1,
                          log_to_console=True,python_log_level='debug')
    trainer.fit()
    # run_name = trainer.state.run_name
    # print(run_name)
    # trainer.fit()
    # trainer.close()
    # # storage_reader = dist_cp.FileSystemReader(f"./test_checkpoints/{run_name}/ba2")
    # # md = storage_reader.read_metadata()
    # # print(md)
    # # # # # ## Load
    # trainer2 = get_trainer(
    #     dataset,
    #     dataloader,
    #     save_folder=s3_folder,
    #     num_features=num_features,
    #     autoresume=True,
    #     run_name='ar-testy-test3',
    #     num_classes=num_classes,
    #     fsdp_state_dict_type='sharded',
    #     max_duration='4ba',
    #     load_weights_only=False,
    #     #load_path=str(pathlib.Path(local_folder.format(run_name=run_name)) / pathlib.Path('ba2')),
    #     log_to_console=True,
    #     python_log_level='debug'
    # )
    # trainer2.fit()
