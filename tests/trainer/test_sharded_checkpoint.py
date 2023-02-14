import torch

from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist
import os
import pytest
from tests.common.markers import world_size
from tests.common import SimpleModel, RandomClassificationDataset
import pathlib


def get_trainer(save_folder: str, save_filename: str, num_features=2, fsdp_state_dict_type='full'):
    model = SimpleModel(num_features=num_features)
    dataset = RandomClassificationDataset(shape=(num_features, 1, 1), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=32)
    optim = torch.optim.Adam(params=model.parameters())
    trainer = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={'min_params': 16,
                    'state_dict_type': fsdp_state_dict_type,
                    'sharding_strategy': 'FULL_SHARD'
                    },
        save_folder=save_folder,
        max_duration='2ba',
        save_interval='2ba',
        save_filename=save_filename,
        progress_bar=False,
        log_to_console=False,

    )
    return trainer

@pytest.mark.gpu
@world_size(2)
def test_fsdp_full_state_dict_save(world_size, tmp_path: pathlib.Path):
    
    
    save_folder=tmp_path
    save_filename='rank{rank}.pt'
    trainer = get_trainer(save_folder=str(save_folder),
                            save_filename=save_filename,
                            fsdp_state_dict_type='full')
    with pytest.warns():
        trainer.fit()
    rankn_checkpoint = save_folder / pathlib.Path(f'rank{dist.get_global_rank()}.pt')
    if dist.get_global_rank() == 0:
        assert os.path.exists(rankn_checkpoint)
    elif dist.get_global_rank() == 1:
        assert not os.path.exists(rankn_checkpoint)
    state_dict_in_memory = trainer.state.state_dict()
    if dist.get_global_rank() == 0:

        with open(str(rankn_checkpoint), 'rb') as f:
            state_dict_from_checkpoint = torch.load(f)

        # Check that model params are equal between in memory mode and checkpoint
        model_params_from_checkpoint = state_dict_from_checkpoint['state']['model']
        model_params_from_memory = state_dict_in_memory['model']
        for param_name in model_params_from_memory.keys():
            cp_model_tensor = model_params_from_checkpoint[param_name]
            mem_model_tensor = model_params_from_memory[param_name]
            assert torch.equal(cp_model_tensor, mem_model_tensor), f"Weight named {param_name} not the same between model checkpoint and in memory model"

        # Check that optim params are equal between checkpoint and in memory optimizer
        optim_params_from_checkpoint = state_dict_from_checkpoint['state']['optimizers']['Adam']['state']
        optim_params_from_memory = state_dict_in_memory['optimizers']['Adam']['state']
        for param_name in optim_params_from_memory.keys():
            cp_param_moment_dict = optim_params_from_checkpoint[param_name]
            mem_param_moment_dict = optim_params_from_memory[param_name]
            for moment_name in mem_param_moment_dict.keys():
                cp_moment = cp_param_moment_dict[moment_name]
                mem_moment = mem_param_moment_dict[moment_name]
                assert torch.equal(cp_moment, mem_moment), f"Moment {moment_name} for parameter {param_name} not the same between optimizer checkpoint and in memory optimizer"
