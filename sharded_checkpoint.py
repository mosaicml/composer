# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import torch

from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist
#from tests.common import RandomClassificationDataset, SimpleModel
# from tests.common.markers import world_size



from composer.models import ComposerClassifier
from typing import Sequence
from torch.utils.data import DataLoader, Dataset

from composer.utils import dist


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 1, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, 5)
        fc2 = torch.nn.Linear(5, num_classes)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = fc1
        self.fc2 = fc2



def get_trainer(save_folder=None,
                save_filename='ba{batch}-rank{rank}.pt',
                num_features=2,
                num_classes=2,
                fsdp_state_dict_type='full',
                load_path=None):
    model = SimpleModel(num_features=num_features, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_features, 1, 1), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=32)
    optim = torch.optim.Adam(params=model.parameters())
    trainer = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={
            'min_params': 16,
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': 'FULL_SHARD'
        },
        save_folder=save_folder,
        max_duration='2ba',
        save_interval='2ba',
        save_filename=save_filename,
        load_path=load_path,
        progress_bar=False,
        log_to_console=False,
    )
    return trainer


def test_fsdp_full_downloads_one_file_only(tmp_path: pathlib.Path = '.'):
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    num_features = 3
    num_classes = 2
    download_folder = os.path.join(tmp_path, f'rank_{dist.get_global_rank()}_download_folder')
    trainer = get_trainer(save_folder=str(save_folder),
                          save_filename=save_filename,
                          num_features=num_features,
                          num_classes=num_classes,
                          fsdp_state_dict_type='full')

    trainer.fit()
    trainer.close()
    load_path = str(save_folder / pathlib.Path('rank0.pt'))
    trainer2 = get_trainer(fsdp_state_dict_type='full', load_path=load_path)

    # with monkeypatch.context() as m:
    #     m.setattr(tempfile, 'TemporaryDirectory', partial(contextlib.nullcontext, enter_result=download_folder))
    #     spoof_get_global_rank = MagicMock(return_value=0)
    #     spoof_get_local_rank = MagicMock(return_value=0)
    #     m.setattr(dist, 'get_global_rank', spoof_get_global_rank)
    #     m.setattr(dist, 'get_local_rank', spoof_get_local_rank)
    #     load_path = str(save_folder / pathlib.Path('rank0.pt'))
    #     trainer2 = get_trainer(fsdp_state_dict_type='full', load_path=load_path)
    #     trainer2.close()
    
    # downloaded_chkpt_path = os.path.join(download_folder, 'rank0_checkpoint' )
    # assert not os.path.exists(downloaded_chkpt_path)
        
    # else:
    #     with monkeypatch.context() as m:
    #         m.setattr(tempfile, 'TemporaryDirectory', partial(contextlib.nullcontext, enter_result=download_folder))
    #         load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    #         get_trainer(fsdp_state_dict_type='full', load_path=load_path)
    #         dist.barrier()
    #     downloaded_chkpt_path = os.path.join(download_folder, 'rank0_checkpoint' )
    #     assert not os.path.exists(downloaded_chkpt_path)

if __name__ == '__main__':
    test_fsdp_full_downloads_one_file_only()
