# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import os
import pathlib
import shutil
import tarfile
import tempfile
import time
from glob import glob
from typing import Any, Dict, List, Optional, Union

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader

from composer.algorithms import SqueezeExcite
from composer.core.callback import Callback
from composer.core.time import Time, TimeUnit
from composer.loggers import ObjectStoreLogger
from composer.optim import ExponentialScheduler
from composer.trainer.trainer import Trainer
from composer.utils import dist, is_tar
from composer.utils.checkpoint import glob_filter
from composer.utils.object_store.libcloud_object_store import LibcloudObjectStore
from tests.common import RandomImageDataset, SimpleConvModel, deep_compare, device


class DummyStatefulCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.random_value = time.time_ns()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'random_value': self.random_value,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.random_value = state['random_value']


def _load_checkpoint(filename: Union[str, pathlib.Path]):
    filename = str(filename).format(rank=0)
    if not is_tar(filename):
        return torch.load(filename, map_location='cpu')

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(filename) as tarball:
            tarball.extractall(tmp_dir)
        states_path = os.path.join(tmp_dir, 'composer_states.pt')
        return torch.load(states_path, map_location='cpu')


@pytest.mark.parametrize(
    'remove_field_paths,filter_params',
    [
        [[['state', 'model', 'classifier', 'weights'], ['state', 'model', 'classifier', 'bias']],
         ['state/model/classifier/weights', 'state/model/classifier/bias']],
        [[
            ['state', 'model', 'classifier', 'weights'],
            ['state', 'model', 'classifier', 'bias'],
        ], ['state/model/classifier/*']],
        [
            [['state', 'timestep']],
            ['state/timestep'],
        ],
        [
            [['state', 'list_element', 0]],
            ['state/list_element/0'],
        ],
        [
            [['state', 'list_element', 0, 'nested_list_element']],
            ['state/list_element/0/nested_list_element'],
        ],
        [
            # Repeating the zero for the test case as removing a list index shifts everything
            [['state', 'list_element', 0], ['state', 'list_element', 0]],
            ['state/list_element/0', 'state/list_element/1'],
        ],
        [
            [
                ['state', 'model', 'classifier', 'weights'],
                ['state', 'model', 'layer1', 'weights'],
                ['state', 'model', 'layer2', 'weights'],
            ],
            ['state/model/*/weights'],
        ],
        [
            [['state', 'model', 'layer1', 'weights'], ['state', 'model', 'layer2', 'weights']],
            ['state/model/layer*/weights'],
        ],
    ],
)
def test_ignore_params(remove_field_paths: List[List[str]], filter_params: List[str]):
    # Set up base dictionary
    base_dict = {
        'state': {
            'run_name': 'my_first_run',
            'timestep': 7,
            'list_element': [{
                'nested_list_element': 'hello'
            }, 'world'],
            'model': {
                'layer1': {
                    'weights': 6,
                    'bias': 2
                },
                'layer2': {
                    'weights': 7,
                    'bias': 1
                },
                'classifier': {
                    'weights': 5,
                    'bias': 3
                }
            }
        },
        'rng': 0,
    }

    # Remove classifier params
    new_dict = copy.deepcopy(base_dict)
    for remove_field_path in remove_field_paths:
        temp_dict = base_dict
        for step in remove_field_path[:-1]:
            temp_dict = temp_dict[step]
        del temp_dict[remove_field_path[-1]]

    glob_filter(filter_params)(new_dict)
    assert base_dict == new_dict


class TestCheckpointLoading:

    def _assert_weights_equivalent(self, m1: torch.nn.Module, m2: torch.nn.Module):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)

    def get_trainer(self, **kwargs):
        model = SimpleConvModel()
        optimizer = torch.optim.Adam(model.parameters())

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=RandomImageDataset(),
                batch_size=8,
            ),
            eval_dataloader=DataLoader(
                dataset=RandomImageDataset(),
                batch_size=16,
            ),
            grad_accum=2,
            precision='fp32',
            train_subset_num_batches=5,
            save_interval='1ep',
            eval_interval='1ba',
            save_filename='ep{epoch}.pt',
            max_duration='2ep',
            optimizers=optimizer,
            schedulers=ExponentialScheduler(gamma=0.9),
            callbacks=[DummyStatefulCallback()],
            **kwargs,
        )

    def get_logger(self, tmp_path: pathlib.Path):
        """Returns an object store logger that saves locally."""
        remote_dir = str(tmp_path / 'object_store')
        os.makedirs(remote_dir, exist_ok=True)

        return ObjectStoreLogger(
            object_store_cls=LibcloudObjectStore,
            object_store_kwargs={
                'provider': 'local',
                'container': '.',
                'provider_kwargs': {
                    'key': remote_dir,
                },
            },
            num_concurrent_uploads=1,
            use_procs=False,
            upload_staging_folder=str(tmp_path / 'staging_folder'),
        )

    @device('cpu', 'gpu')
    @pytest.mark.parametrize('load_weights_only', [True, False])
    def test_load_weights(self, device, load_weights_only):

        trainer_1 = self.get_trainer(save_folder='first', device=device)
        trainer_1.fit()
        trainer_1.close()

        last_checkpoint = os.path.join('first', 'ep2.pt')
        trainer_2 = self.get_trainer(
            load_path=last_checkpoint,
            load_weights_only=load_weights_only,
            load_strict_model_weights=load_weights_only,
        )

        # check weights loaded properly
        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

        # check callbacks state
        stateful_callbacks_equal = self._stateful_callbacks_equal(
            trainer_1.state.callbacks,
            trainer_2.state.callbacks,
        )
        if load_weights_only:
            # callback state should not have been loaded
            assert not stateful_callbacks_equal
        else:
            assert stateful_callbacks_equal

    def _stateful_callbacks_equal(self, callbacks1, callbacks2):

        cb1 = next((cb for cb in callbacks1 if isinstance(cb, DummyStatefulCallback)))
        cb2 = next((cb for cb in callbacks2 if isinstance(cb, DummyStatefulCallback)))

        return cb1.random_value == cb2.random_value

    def test_load_weights_object_store(self, tmp_path):

        pytest.importorskip('libcloud')

        trainer_1 = self.get_trainer(
            save_folder='first',
            loggers=[self.get_logger(tmp_path)],
            run_name='electric-zebra',
        )
        trainer_1.fit()
        trainer_1.close()

        trainer_2 = self.get_trainer(
            loggers=[self.get_logger(tmp_path)],
            run_name='electric-zebra',
            load_path='electric-zebra/checkpoints/latest-rank0',
            load_object_store=self.get_logger(tmp_path),
        )

        # check weights loaded properly
        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

    @device('cpu', 'gpu')
    @pytest.mark.parametrize('use_object_store', [True, False])
    @pytest.mark.parametrize('delete_local', [True, False])
    def test_autoresume(
        self,
        device: str,
        tmp_path: pathlib.Path,
        use_object_store: bool,
        delete_local: bool,
    ):
        if delete_local and not use_object_store:
            pytest.skip('Invalid test setting.')

        if use_object_store:
            pytest.importorskip('libcloud')

        trainer_1 = self.get_trainer(
            save_folder='first',
            device=device,
            run_name='big-chungus',
            loggers=[self.get_logger(tmp_path)] if use_object_store else [],
        )

        # trains the model, saving the checkpoint files
        trainer_1.fit()
        trainer_1.close()

        if delete_local:
            # delete files locally, forcing trainer to look in object store
            shutil.rmtree('first')

        trainer_2 = self.get_trainer(
            save_folder='first',
            device=device,
            run_name='big-chungus',
            autoresume=True,
            load_path='ignore_me.pt',  # this should be ignored
            loggers=[self.get_logger(tmp_path)] if use_object_store else [],
        )

        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

        assert trainer_1.state.run_name == trainer_2.state.run_name

    def test_different_run_names(self):

        trainer_1 = self.get_trainer(
            save_folder='first/',
            seed=12345,
        )
        trainer_1.fit()
        trainer_1.close()

        trainer_2 = self.get_trainer(
            load_path=os.path.join('first', 'ep2.pt'),
            seed=12345,
        )

        assert trainer_1.state.run_name != trainer_2.state.run_name

    @device('cpu', 'gpu')
    @pytest.mark.parametrize('save_overwrite', [True, False])
    def test_save_overwrite(self, device, save_overwrite):

        trainer_1 = self.get_trainer(
            save_folder='first',
            device=device,
        )
        trainer_1.fit()
        trainer_1.close()

        ctx = None
        if save_overwrite:
            ctx = contextlib.nullcontext()
        else:
            ctx = pytest.raises(FileExistsError)

        with ctx:  # expect FileExistsError if save_overwrite=False
            trainer_2 = self.get_trainer(
                save_folder='first',
                save_overwrite=save_overwrite,
                load_path=os.path.join('first', 'ep1.pt'),
                device=device,
            )
            trainer_2.fit(duration='1ba')

        # loading from the last checkpoint should work regardless
        # of save_overwrite, as new checkpoints are later in time.
        trainer_3 = self.get_trainer(
            save_folder='first',
            save_overwrite=save_overwrite,
            load_path=os.path.join('first', 'ep2.pt'),
            device=device,
        )
        trainer_3.fit(duration='1ba')

    def test_surgery_resumption(self, tmp_path: pathlib.Path):
        trainer_1 = self.get_trainer(
            algorithms=[SqueezeExcite(latent_channels=64, min_channels=3)],
            save_folder=os.path.join(tmp_path, 'first'),
        )
        trainer_1.fit()
        trainer_1.close()

        # ValueError is raised without surgery
        resume_file = os.path.join(tmp_path, 'first', 'ep1.pt')
        with pytest.raises(ValueError) as e:
            self.get_trainer(load_path=resume_file)
        assert 'The following surgery algorithms' in str(e)

        # Loads fine with surgery
        self.get_trainer(
            algorithms=[SqueezeExcite(latent_channels=64, min_channels=3)],
            load_path=resume_file,
        )


class TestCheckpointResumption:

    def get_trainer(self, **kwargs):
        model = SimpleConvModel()
        optimizer = torch.optim.Adam(model.parameters())

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=RandomImageDataset(),
                batch_size=8,
                shuffle=False,
            ),
            eval_dataloader=DataLoader(
                dataset=RandomImageDataset(),
                batch_size=16,
                shuffle=False,
            ),
            grad_accum=2,
            precision='fp32',
            train_subset_num_batches=5,
            max_duration='2ep',
            optimizers=optimizer,
            schedulers=ExponentialScheduler(gamma=0.9),
            callbacks=[DummyStatefulCallback()],
            **kwargs,
        )

    @pytest.mark.parametrize('world_size', [
        pytest.param(1),
        pytest.param(2, marks=pytest.mark.world_size(2)),
    ])
    @pytest.mark.parametrize('device,deepspeed_zero_stage', [
        pytest.param('cpu', None, id='cpu-ddp'),
        pytest.param('gpu', None, id='gpu-ddp', marks=pytest.mark.gpu),
        pytest.param('gpu', 0, id='deepspeed-zero0', marks=pytest.mark.gpu),
        pytest.param('gpu', 1, id='deepspeed-zero1', marks=pytest.mark.gpu),
        pytest.param('gpu', 2, id='deepspeed-zero2', marks=pytest.mark.gpu),
    ])
    @pytest.mark.parametrize(
        'seed,save_interval,save_filename,resume_file,final_checkpoint',
        [
            [None, '1ep', 'ep{epoch}-rank{rank}.pt', 'ep1-rank{rank}.pt', 'latest-rank{rank}.pt'
            ],  # test randomized seed saving and symlinking
            [42, '1ep', 'ep{epoch}-rank{rank}.pt', 'ep1-rank{rank}.pt', 'ep2-rank{rank}.pt'],  # test save at epoch end
            [42, '1ep', 'ep{epoch}-rank{rank}.tgz', 'ep1-rank{rank}.tgz', 'ep2-rank{rank}.tgz'
            ],  # test tarball with compression
            [42, '2ba', 'ba{batch}-rank{rank}.pt', 'ba4-rank{rank}.pt', 'ba8-rank{rank}.pt'
            ],  # test save batch in partial epoch
            [42, '1ba', 'ba{batch}-rank{rank}.pt', 'ba5-rank{rank}.pt', 'ba8-rank{rank}.pt'
            ],  # test save batch at epoch end
            [42, '2ba', 'ba{batch}-rank{rank}.pt', 'ba6-rank{rank}.pt', 'ba8-rank{rank}.pt'
            ],  # test save batch after complete epoch
        ],
    )
    def test_resumption(
        self,
        device: str,
        world_size: int,
        deepspeed_zero_stage: Optional[int],
        save_interval: str,
        save_filename: str,
        resume_file: str,
        final_checkpoint: str,
        seed: Optional[int],
        tmp_path: pathlib.Path,
    ):

        # all ranks use rank 0 folder
        tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
        save_folder = pathlib.Path(tmp_paths[0])

        if deepspeed_zero_stage:
            deepspeed_config = {'zero_optimization': {'stage': deepspeed_zero_stage}}

            # save_checkpoint appends .tar for deepspeed
            if not is_tar(resume_file):
                resume_file += '.tar'
            if not is_tar(final_checkpoint):
                final_checkpoint += '.tar'
        else:
            deepspeed_config = None

        trainer_1 = self.get_trainer(
            save_folder=os.path.join(save_folder, 'first'),
            save_filename=save_filename,
            save_interval=save_interval,
            eval_interval=save_interval,
            deepspeed_config=deepspeed_config,
            seed=seed,
            device=device,
        )

        trainer_1.fit()
        trainer_1.close()

        self._assert_expected_num_checkpoints(
            save_folder=os.path.join(save_folder, 'first'),
            save_interval=save_interval,
            num_epochs=2,  # set in get_trainer()
            num_batches_per_epoch=5,  # set in get_trainer()
            is_deepspeed=deepspeed_config is not None,
        )

        if not deepspeed_config:
            # for DDP training, only rank 0 saves
            resume_file = resume_file.format(rank=0)

        resume_file = os.path.join(save_folder, 'first', resume_file)

        trainer_2 = self.get_trainer(
            save_folder=os.path.join(save_folder, 'second'),
            save_filename=save_filename,
            save_interval=save_interval,
            eval_interval=save_interval,
            deepspeed_config=deepspeed_config,
            seed=seed,
            device=device,
            load_path=resume_file,  # <-- resume training from file
        )
        trainer_2.fit()
        trainer_2.close()

        self._assert_checkpoints_equivalent(
            save_folder / 'first' / final_checkpoint,
            save_folder / 'second' / final_checkpoint,
        )

    def _assert_checkpoints_equivalent(self, file1, file2):
        checkpoint_1 = _load_checkpoint(file1)
        checkpoint_2 = _load_checkpoint(file2)

        # Remove the wall clock time
        del checkpoint_1['state']['timestamp']['Timestamp']['total_wct']
        del checkpoint_1['state']['timestamp']['Timestamp']['epoch_wct']
        del checkpoint_1['state']['timestamp']['Timestamp']['batch_wct']
        del checkpoint_2['state']['timestamp']['Timestamp']['total_wct']
        del checkpoint_2['state']['timestamp']['Timestamp']['epoch_wct']
        del checkpoint_2['state']['timestamp']['Timestamp']['batch_wct']

        # Remove run_name, since it's a function of time
        del checkpoint_1['state']['run_name']
        del checkpoint_2['state']['run_name']

        deep_compare(checkpoint_1, checkpoint_2)

        # deepspeed checkpoints do not have model or optimizer
        # so either model, optimizer should be in all checkpoints or in none
        keys_in = (
            'model' in checkpoint_1['state'],
            'optimizers' in checkpoint_1['state'],
            'model' in checkpoint_2['state'],
            'optimizers' in checkpoint_2['state'],
        )
        assert all(keys_in) or not any(keys_in)

    def _assert_expected_num_checkpoints(
        self,
        save_folder: str,
        save_interval: str,
        num_epochs: int,
        num_batches_per_epoch: int,
        is_deepspeed: bool,
    ):
        interval = Time.from_timestring(save_interval)
        if interval.unit == TimeUnit.EPOCH:
            expected_num_files = ((num_epochs - 1) // interval.value) + 1
        else:
            expected_num_files = ((num_batches_per_epoch * num_epochs - 1) // interval.value) + 1
        expected_num_files += 1  # account for symlink

        if is_deepspeed:
            # each rank saves
            expected_num_files *= dist.get_world_size()

        files = os.listdir(save_folder)
        assert len(files) == expected_num_files


@pytest.mark.parametrize('world_size', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize('device,deepspeed_enabled,zero_stage', [
    pytest.param('cpu', False, None, id='cpu-ddp'),
    pytest.param('gpu', False, None, id='gpu-ddp', marks=pytest.mark.gpu),
    pytest.param('gpu', True, 0, id='deepspeed-zero0', marks=pytest.mark.gpu),
    pytest.param('gpu', True, 1, id='deepspeed-zero1', marks=pytest.mark.gpu),
    pytest.param('gpu', True, 2, id='deepspeed-zero2', marks=pytest.mark.gpu),
])
def test_rotate_checkpoints(
    world_size,
    device,
    deepspeed_enabled,
    zero_stage,
    tmp_path: pathlib.Path,
):
    num_keep = 5

    # all ranks use rank 0 folder
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = tmp_paths[0]

    if deepspeed_enabled:
        deepseed_config = {'zero_optimization': {'stage': zero_stage}}
    else:
        deepseed_config = None

    trainer = Trainer(
        model=SimpleConvModel(),
        train_dataloader=DataLoader(dataset=RandomImageDataset()),
        save_folder=str(save_folder),
        save_filename='checkpoint_{rank}_{batch}.pt',
        save_interval='1ba',
        max_duration='10ba',
        save_num_checkpoints_to_keep=num_keep,
        device=device,
        deepspeed_config=deepseed_config,
    )

    trainer.fit()

    dist.barrier()  # ensure all checkpoints rotated across ranks

    # deepspeed saves 1 file per rank
    expected_num = num_keep if not deepspeed_enabled else num_keep * world_size

    files = glob(os.path.join(save_folder, 'checkpoint_*'))
    assert len(files) == expected_num

    dist.barrier()  # all ranks finish before cleaning up tmpdir
