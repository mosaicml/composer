import torch
from torch.utils.data import DataLoader, Dataset
from typing import Sequence
from packaging import version
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from composer.models import ComposerClassifier
import torch.distributed
import pytest
import pathlib
import textwrap
import os
from tests.common.markers import world_size

def _compare_optims_between_state_dicts(state_dict1, state_dict2):
    # Check that optim params are equal between checkpoint and in memory optimizer
    state_dict1_optim_params = state_dict1['optimizers']['state']
    state_dict2_optim_params = state_dict2['optimizers']['state']
    state_dict1_keys = set(state_dict1_optim_params.keys())
    state_dict2_keys = set(state_dict2_optim_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""")

    for param_name in state_dict2_optim_params.keys():
        state_dict1_param_moment_dict = state_dict1_optim_params[param_name]
        state_dict2_param_moment_dict = state_dict2_optim_params[param_name]
        for moment_name in state_dict2_param_moment_dict.keys():
            state_dict1_moment = state_dict1_param_moment_dict[moment_name]
            state_dict2_moment = state_dict2_param_moment_dict[moment_name]
            assert torch.equal(
                state_dict1_moment,
                state_dict2_moment), f'Moment {moment_name} for parameter {param_name} not the same between state dicts'


def _compare_model_params_between_state_dicts(state_dict1, state_dict2):
    # Check that model params are equal between in memory mode and checkpoint
    state_dict1_model_params = state_dict1['model']
    state_dict2_model_params = state_dict2['model']

    state_dict1_keys = set(state_dict1_model_params.keys())
    state_dict2_keys = set(state_dict2_model_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys that belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""")

    for param_name in state_dict2_model_params.keys():
        state_dict1_model_tensor = state_dict1_model_params[param_name]
        state_dict2_model_tensor = state_dict2_model_params[param_name]
        assert torch.equal(state_dict1_model_tensor,
                           state_dict2_model_tensor), f'Weight named {param_name} not the same between state_dicts'
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



def get_trainer(dataset,
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
        save_interval='2ba',
        save_filename=save_filename,
        save_overwrite=False,
        save_weights_only=save_weights_only,
        load_path=load_path,
        load_weights_only=load_weights_only,
        progress_bar=False,
        log_to_console=log_to_console,
        autoresume=autoresume,
        run_name=run_name,
        python_log_level=python_log_level,
        save_latest_filename=None,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
    )
    return trainer

# if __name__ == '__main__':

@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['sharded', 'local'])
@pytest.mark.parametrize('autoresume', [False])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_partitioned_state_dict_load(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool):
    from torch.distributed import checkpoint as dist_cp
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    import time
    from composer.core.state import fsdp_state_dict_type_context
    num_features = 16
    num_classes = 8
    dataset = RandomClassificationDataset(shape=(num_features, 1, 1), num_classes=num_classes, size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=16)
    ## Save
    save_path = dist.all_gather_object(tmp_path)[0]
    s3_folder = 's3://mosaicml-internal-checkpoints-test/evan-test/test_sharded_checkpoints/{run_name}'
    local_folder = str(save_path / pathlib.Path('{run_name}'))
    local_copy_of_s3_folder = './evan-test/test_sharded_checkpoints/{run_name}'
    trainer1 = get_trainer(dataset,
                          dataloader,
                          num_features=num_features,
                          num_classes=num_classes,
                          save_folder=local_folder,
                          save_weights_only=False,
                          max_duration='2ba',
                          fsdp_state_dict_type='sharded',
                          save_num_checkpoints_to_keep=-1,
                          log_to_console=False,
                          )
    run_name = trainer1.state.run_name
    print(run_name)
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    #assert os.listdir(local_folder.format(run_name=run_name) + '/ba2') == ['foo']


    # # # # ## Load
    trainer2 = get_trainer(dataset,
                           dataloader,
                           num_features=num_features,
                           num_classes=num_classes,
                           fsdp_state_dict_type='sharded',
                           max_duration='2ba',
                           load_weights_only=False,
                           load_path=local_folder.format(run_name=run_name) + '/ba2',
                           log_to_console=False,
                           )
    state_dict_from_trainer2 = trainer2.state.state_dict()
    #Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
    #trainer2.fit()
    #print(trainer2.state.state_dict()['model']['module.2.weight'].local_tensor())
    # sd = {'model' : trainer2.state.state_dict()['model']}
    # storage_reader  = dist_cp.FileSystemReader(f"./test_checkpoints/{run_name}/ba2")

    # # dist_cp.load_state_dict(sd, storage_reader)
    # print(trainer2.state._optimizer_state_dict()['state']['module.2.weight']['exp_avg'].local_tensor())
    # # trainer2.state.load_model_state(sd, trainer2.logger, strict=True)
    # optim_state = load_sharded_optimizer_state_dict(
    #     model_state_dict=trainer2.state.state_dict()['model'],
    #     optimizer_key="optimizers",
    #     storage_reader=storage_reader,
    # )
    # print(optim_state['optimizers']['state']['module.2.weight']['exp_avg'].local_tensor())
    # trainer2.state.load_optim_state(optim_state)
    # print(trainer2.state._optimizer_state_dict()['state']['module.2.weight']['exp_avg'].local_tensor())
