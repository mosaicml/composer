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
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed
from packaging import version
from pytest import MonkeyPatch
from torch.utils.data import DataLoader

from composer.algorithms import NoOpModel
from composer.callbacks import CheckpointSaver
from composer.core import Callback, Time, TimeUnit
from composer.loggers import RemoteUploaderDownloader, remote_uploader_downloader
from composer.optim import ExponentialScheduler
from composer.trainer import trainer
from composer.trainer.trainer import Trainer
from composer.utils import dist, is_tar
from composer.utils.checkpoint import glob_filter
from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore
from tests.common import (RandomClassificationDataset, RandomImageDataset, RandomTextLMDataset, SimpleConvModel,
                          SimpleModel, SimpleTransformerMaskedLM, deep_compare, device)
from tests.common.markers import world_size


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


def _assert_checkpoints_equivalent(file1, file2, atol=0.0, rtol=0.0):
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

    # Remove dummy stateful callback
    for ckpt in [checkpoint_1, checkpoint_2]:
        if 'DummyStatefulCallback' in ckpt['state']['callbacks']:
            del ckpt['state']['callbacks']['DummyStatefulCallback']

    deep_compare(checkpoint_1, checkpoint_2, atol=atol, rtol=rtol)

    # deepspeed checkpoints do not have model or optimizer
    # so either model, optimizer should be in all checkpoints or in none
    keys_in = (
        'model' in checkpoint_1['state'],
        'optimizers' in checkpoint_1['state'],
        'model' in checkpoint_2['state'],
        'optimizers' in checkpoint_2['state'],
    )
    assert all(keys_in) or not any(keys_in)


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


@pytest.mark.parametrize('folder,filename',
                         [('{run_name}/my_checkpoints', 'ep{epoch}-rank{rank}.pt'),
                          (pathlib.Path('{run_name}/my_checkpoints'), pathlib.Path('ep{epoch}-rank{rank}.pt'))])
def test_checkpoint_saver_folder_filename_path(folder: Union[str, pathlib.Path], filename: Union[str, pathlib.Path]):
    checkpoint_saver = CheckpointSaver(folder=folder, filename=filename)

    assert checkpoint_saver.folder == str(folder)
    assert checkpoint_saver.filename.filename == str(filename)


@pytest.mark.parametrize(
    'remote_file_name,latest_filename,latest_remote_file_name',
    [('{run_name}/my_checkpoints/ep{epoch}-ba{batch}-rank{rank}.pt', 'latest-rank{rank}.pt',
      '{run_name}/checkpoints/latest-rank{rank}.pt'),
     (pathlib.Path('{run_name}/my_checkpoints/ep{epoch}-ba{batch}-rank{rank}.pt'), pathlib.Path('latest-rank{rank}.pt'),
      pathlib.Path('{run_name}/checkpoints/latest-rank{rank}.pt'))])
def test_checkpoint_filenames(remote_file_name: Optional[Union[str, pathlib.Path]],
                              latest_filename: Optional[Union[str, pathlib.Path]],
                              latest_remote_file_name: Optional[Union[str, pathlib.Path]]):
    checkpoint_saver = CheckpointSaver(remote_file_name=remote_file_name,
                                       latest_filename=latest_filename,
                                       latest_remote_file_name=latest_remote_file_name)

    assert checkpoint_saver.remote_file_name is not None
    assert checkpoint_saver.latest_filename is not None
    assert checkpoint_saver.latest_remote_file_name is not None

    assert checkpoint_saver.remote_file_name.filename == str(remote_file_name)
    assert checkpoint_saver.latest_filename.filename == str(latest_filename)
    assert checkpoint_saver.latest_remote_file_name.filename == str(latest_remote_file_name)


@pytest.mark.parametrize('remote_file_name,latest_filename,latest_remote_file_name', [(None, None, None)])
def test_checkpoint_filenames_none(remote_file_name: Optional[Union[str, pathlib.Path]],
                                   latest_filename: Optional[Union[str, pathlib.Path]],
                                   latest_remote_file_name: Optional[Union[str, pathlib.Path]]):
    checkpoint_saver = CheckpointSaver(remote_file_name=remote_file_name,
                                       latest_filename=latest_filename,
                                       latest_remote_file_name=latest_remote_file_name)

    assert checkpoint_saver.remote_file_name == None
    assert checkpoint_saver.latest_filename == None
    assert checkpoint_saver.latest_remote_file_name == None


class TestCheckpointSaving:

    def get_trainer(self, **kwargs):
        model = SimpleConvModel()
        return Trainer(model=model, **kwargs)

    @pytest.mark.parametrize('add_remote_ud', [True, False])
    def test_s3_uri_creates_remote_ud(self, add_remote_ud: bool, monkeypatch: MonkeyPatch):
        mock_validate_credentials = MagicMock()
        monkeypatch.setattr(remote_uploader_downloader, '_validate_credentials', mock_validate_credentials)
        if add_remote_ud:
            with pytest.warns(UserWarning):
                trainer = self.get_trainer(save_folder='s3://bucket_name/{run_name}/checkpoints',
                                           loggers=[
                                               RemoteUploaderDownloader('s3://bucket_name',
                                                                        file_path_format_string='{remote_file_name}')
                                           ])
        else:
            trainer = self.get_trainer(save_folder='s3://bucket_name/{run_name}/checkpoints')

        remote_uds = [
            logger_dest for logger_dest in trainer.logger.destinations
            if isinstance(logger_dest, RemoteUploaderDownloader)
        ]
        assert len(remote_uds) == 1
        remote_ud = remote_uds[0]
        assert remote_ud.remote_backend_name == 's3'
        assert remote_ud.remote_bucket_name == 'bucket_name'

    @pytest.mark.parametrize('uri', ['wandb://foo/bar', 'gcs://foo/bar', 'sftp://foo/bar"'])
    def test_other_uris_error_out(self, uri: str):
        with pytest.raises(NotImplementedError):
            self.get_trainer(save_folder=uri)

    @pytest.mark.parametrize('local_path', ['foo/bar/baz'])
    def test_local_paths_work(self, local_path: str):
        self.get_trainer(save_folder=local_path)

    @pytest.mark.parametrize('save_folder,expected_path',
                             [('s3://bucket_name/{run_name}/my_checkpoints', '{run_name}/my_checkpoints'),
                              ('{run_name}/my_checkpoints', '{run_name}/my_checkpoints'), ('s3://bucket_name', '')])
    def test_checkpoint_saver_properly_constructed(self, save_folder: str, expected_path: str,
                                                   monkeypatch: MonkeyPatch):
        mock_validate_credentials = MagicMock()
        monkeypatch.setattr(remote_uploader_downloader, '_validate_credentials', mock_validate_credentials)
        mock_checkpoint_saver = MagicMock()
        monkeypatch.setattr(trainer, 'CheckpointSaver', mock_checkpoint_saver)
        self.get_trainer(save_folder=save_folder)
        expected_prefix = expected_path + '/' if expected_path != '' else expected_path
        rest_of_checkpoint_saver_kwargs = {
            'filename': 'ep{epoch}-ba{batch}-rank{rank}.pt',
            'remote_file_name': expected_prefix + 'ep{epoch}-ba{batch}-rank{rank}.pt',
            'latest_filename': 'latest-rank{rank}.pt',
            'latest_remote_file_name': expected_prefix + 'latest-rank{rank}.pt',
            'overwrite': False,
            'weights_only': False,
            'save_interval': '1ep',
            'num_checkpoints_to_keep': -1,
        }
        expected_folder = expected_path.rstrip('/') if expected_path != '' else '.'
        mock_checkpoint_saver.assert_called_once_with(folder=expected_folder, **rest_of_checkpoint_saver_kwargs)

    @pytest.mark.parametrize('save_interval', ['1tok', '64tok', '65tok'])
    @pytest.mark.parametrize('batch_size', [1, 4])
    @pytest.mark.parametrize('sequence_length', [1, 16])
    def test_checkpoint_save_token_interval(self, tiny_bert_tokenizer, save_interval: str, batch_size: int,
                                            sequence_length: int, tmp_path: pathlib.Path):
        tokens_per_batch = batch_size * sequence_length
        max_duration_time = Time.from_timestring('5ba')
        save_interval_time = Time.from_timestring(save_interval)
        max_duration_tokens = max_duration_time.value * tokens_per_batch

        # calculate the expected number of checkpoints
        last_token_iter = 0
        next_multiple = save_interval_time.value
        expected_num_checkpoints = 0
        last_multiple_added = -1
        for token_iter in range(0, max_duration_tokens + tokens_per_batch, tokens_per_batch):
            if last_token_iter < next_multiple <= token_iter:
                last_multiple_added = next_multiple
                expected_num_checkpoints += 1
            last_token_iter = token_iter
            while next_multiple <= last_token_iter:
                next_multiple += save_interval_time.value

        if last_multiple_added + tokens_per_batch <= max_duration_tokens:
            expected_num_checkpoints += 1

        transformers = pytest.importorskip('transformers')
        model = SimpleTransformerMaskedLM(vocab_size=tiny_bert_tokenizer.vocab_size)
        pretraining_train_dataset = RandomTextLMDataset(size=100,
                                                        vocab_size=tiny_bert_tokenizer.vocab_size,
                                                        sequence_length=sequence_length,
                                                        use_keys=True)

        collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
        dataloader = DataLoader(pretraining_train_dataset,
                                batch_size=batch_size,
                                sampler=dist.get_sampler(pretraining_train_dataset),
                                collate_fn=collator)

        trainer = Trainer(model=model,
                          train_dataloader=dataloader,
                          max_duration=max_duration_time,
                          save_interval=save_interval_time,
                          save_folder=str(tmp_path / 'checkpoints'))
        trainer.fit()

        assert trainer._checkpoint_saver is not None
        assert len(trainer._checkpoint_saver.saved_checkpoints) == expected_num_checkpoints

    @pytest.mark.parametrize('save_interval', ['1sp', '4sp', '5sp'])
    @pytest.mark.parametrize('batch_size', [1, 4])
    @pytest.mark.parametrize('sequence_length', [1, 16])
    def test_checkpoint_save_sample_interval(self, tiny_bert_tokenizer, save_interval: str, batch_size: int,
                                             sequence_length: int, tmp_path: pathlib.Path):
        max_duration_time = Time.from_timestring('5ba')
        save_interval_time = Time.from_timestring(save_interval)
        max_duration_samples = max_duration_time.value * batch_size

        # calculate the expected number of checkpoints
        last_sample_iter = 0
        next_multiple = save_interval_time.value
        expected_num_checkpoints = 0
        last_multiple_added = -1
        for sample_iter in range(0, max_duration_samples + batch_size, batch_size):
            if last_sample_iter < next_multiple <= sample_iter:
                last_multiple_added = next_multiple
                expected_num_checkpoints += 1
            last_token_iter = sample_iter
            while next_multiple <= last_token_iter:
                next_multiple += save_interval_time.value

        if last_multiple_added + batch_size <= max_duration_samples:
            expected_num_checkpoints += 1

        transformers = pytest.importorskip('transformers')
        model = SimpleTransformerMaskedLM(vocab_size=tiny_bert_tokenizer.vocab_size)
        pretraining_train_dataset = RandomTextLMDataset(size=100,
                                                        vocab_size=tiny_bert_tokenizer.vocab_size,
                                                        sequence_length=sequence_length,
                                                        use_keys=True)

        collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
        dataloader = DataLoader(pretraining_train_dataset,
                                batch_size=batch_size,
                                sampler=dist.get_sampler(pretraining_train_dataset),
                                collate_fn=collator)

        trainer = Trainer(model=model,
                          train_dataloader=dataloader,
                          max_duration=max_duration_time,
                          save_interval=save_interval_time,
                          save_folder=str(tmp_path / 'checkpoints'))
        trainer.fit()

        assert trainer._checkpoint_saver is not None
        assert len(trainer._checkpoint_saver.saved_checkpoints) == expected_num_checkpoints

    @pytest.mark.parametrize('save_weights_only', [True, False])
    def test_save_weights_only(self, tmp_path: pathlib.Path, save_weights_only: bool):
        model = SimpleConvModel()
        train_dataset = RandomImageDataset()
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=2,
            sampler=dist.get_sampler(train_dataset),
        )
        save_filename = 'ba{batch}-test'
        save_folder = str(tmp_path / 'checkpoints')
        trainer = Trainer(model=model,
                          train_dataloader=train_dataloader,
                          max_duration='1ba',
                          save_folder=save_folder,
                          save_filename=save_filename,
                          save_weights_only=save_weights_only,
                          save_interval='1ba')
        trainer.fit()
        expected_metadata = trainer.state._get_state_metadata()
        expected_integrations = trainer.state._get_integrations_state_dict()
        trainer.close()
        checkpoint_filepath = os.path.join(save_folder, save_filename.format(batch=1))
        composer_state_dict = torch.load(checkpoint_filepath, map_location='cpu')

        if save_weights_only:
            assert set(composer_state_dict['state'].keys()) == {'model', 'metadata', 'integrations'}
            assert composer_state_dict['state']['metadata'] == expected_metadata
            assert composer_state_dict['state']['integrations'] == expected_integrations
        else:
            assert set(composer_state_dict['state'].keys()) != {'model', 'metadata', 'integrations'}


class TestCheckpointLoading:

    def _assert_weights_equivalent(self, m1: torch.nn.Module, m2: torch.nn.Module):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)

    def _metrics_equal(self, train_metrics_1, train_metrics_2, eval_metrics_1, eval_metrics_2):
        try:
            deep_compare(train_metrics_1, train_metrics_2, atol=1e-8, rtol=1e-8)
            deep_compare(eval_metrics_1, eval_metrics_2, atol=1e-7, rtol=1e-7)
            return True
        except AssertionError:
            return False

    def get_trainer(self, model=None, max_duration='2ep', latest_filename='latest-rank{rank}.pt', **kwargs):
        if model is None:
            model = SimpleConvModel()
        optimizer = torch.optim.Adam(model.parameters())

        train_dataset = RandomImageDataset()
        eval_dataset = RandomImageDataset()
        train_batch_size = 2

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                sampler=dist.get_sampler(train_dataset),
            ),
            eval_dataloader=DataLoader(
                dataset=eval_dataset,
                batch_size=4,
                sampler=dist.get_sampler(eval_dataset),
            ),
            device_train_microbatch_size=train_batch_size // 2,
            precision='fp32',
            train_subset_num_batches=5,
            eval_subset_num_batches=1,
            save_interval='1ep',
            eval_interval='1ep',
            save_latest_filename=latest_filename,
            save_filename='ep{epoch}.pt',
            max_duration=max_duration,
            optimizers=optimizer,
            schedulers=ExponentialScheduler(gamma=0.9),
            callbacks=[DummyStatefulCallback()],
            **kwargs,
        )

    def get_logger(self, tmp_path: pathlib.Path):
        """Returns an object store logger that saves locally."""
        remote_dir = str(tmp_path / 'object_store')
        os.makedirs(remote_dir, exist_ok=True)

        return RemoteUploaderDownloader(
            bucket_uri='libcloud://.',
            backend_kwargs={
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

    @world_size(1, 2)
    @device('cpu', 'gpu')
    @pytest.mark.parametrize('use_object_store', [True, False])
    @pytest.mark.parametrize('delete_local', [True, False])
    @pytest.mark.parametrize('test_slashed', [True, False])
    def test_autoresume(self, device: str, tmp_path: pathlib.Path, use_object_store: bool, delete_local: bool,
                        test_slashed: bool, world_size: int):
        if delete_local and not use_object_store:
            pytest.skip('Invalid test setting.')

        if use_object_store:
            pytest.importorskip('libcloud')

        latest_filename = 'latest-rank{rank}.pt'
        if test_slashed:
            latest_filename = 'testdir/' + latest_filename
        trainer_1 = self.get_trainer(
            latest_filename=latest_filename,
            save_folder='first',
            device=device,
            run_name='big-chungus',
            autoresume=True,
            loggers=[self.get_logger(tmp_path)] if use_object_store else [],
        )

        # trains the model, saving the checkpoint files
        trainer_1.fit()
        trainer_1.close()

        if delete_local:
            # delete files locally, forcing trainer to look in object store
            shutil.rmtree('first')

        trainer_2 = self.get_trainer(
            latest_filename=latest_filename,
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

        assert self._metrics_equal(
            trainer_1.state.train_metrics, trainer_2.state.train_metrics, trainer_1.state.eval_metrics,
            trainer_2.state.eval_metrics), 'Original metrics do not equal metrics from loaded checkpoint.'

        assert trainer_1.state.run_name == trainer_2.state.run_name

    @pytest.mark.parametrize('load_path,load_object_store',
                             [('s3://my-bucket/my-run-name/my-checkpoints', None),
                              ('s3://my-bucket/my-run-name/my-checkpoints', S3ObjectStore(bucket='my-bucket')),
                              ('my-run-name/my-checkpoints', S3ObjectStore(bucket='my-bucket'))])
    def test_load_from_uri(self, load_path: str, load_object_store: Optional[ObjectStore], monkeypatch: MonkeyPatch):

        mock_validate_credentials = MagicMock()
        monkeypatch.setattr(remote_uploader_downloader, '_validate_credentials', mock_validate_credentials)
        mock_load_checkpoint = MagicMock()
        monkeypatch.setattr(trainer.checkpoint, 'load_checkpoint', mock_load_checkpoint)
        self.get_trainer(load_path=load_path, load_object_store=load_object_store)
        mock_load_checkpoint.assert_called_once()
        (_, call_kwargs), = mock_load_checkpoint.call_args_list
        assert call_kwargs['path'] == 'my-run-name/my-checkpoints'
        assert isinstance(call_kwargs['object_store'], S3ObjectStore)
        assert call_kwargs['object_store'].bucket == 'my-bucket'

    @pytest.mark.parametrize('load_path', [
        'sftp://my-bucket/my-run-name/my-checkpoints', 'wandb://my-bucket/my-run-name/my-checkpoints',
        'gcs://my-bucket/my-run-name/my-checkpoints'
    ])
    def test_other_backends_error(self, load_path: str, monkeypatch: MonkeyPatch):
        mock_validate_credentials = MagicMock()
        monkeypatch.setattr(remote_uploader_downloader, '_validate_credentials', mock_validate_credentials)
        with pytest.raises(NotImplementedError):
            self.get_trainer(load_path=load_path)

    @pytest.mark.parametrize('missing_key', [True, False])
    @pytest.mark.parametrize('unexpected_key', [True, False])
    def test_strict_errors(self, missing_key: bool, unexpected_key: bool):
        model1 = SimpleConvModel()
        if unexpected_key:
            model1.unexpected_dummy = torch.nn.Parameter(torch.zeros(1))

        trainer_1 = self.get_trainer(model=model1, save_folder='first')
        trainer_1.fit()
        trainer_1.close()

        model2 = SimpleConvModel()
        if missing_key:
            model2.missing_dummy = torch.nn.Parameter(torch.zeros(1))

        last_checkpoint = os.path.join('first', 'ep2.pt')
        if missing_key or unexpected_key:
            error_context = pytest.raises(RuntimeError, match='Failed to load checkpoint due to')
        else:
            error_context = contextlib.nullcontext()

        with error_context:
            _ = self.get_trainer(
                model=model2,
                load_path=last_checkpoint,
                load_weights_only=True,
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

        # check metrics loaded
        metrics_equal = self._metrics_equal(trainer_1.state.train_metrics, trainer_2.state.train_metrics,
                                            trainer_1.state.eval_metrics, trainer_2.state.eval_metrics)

        # check callbacks state
        stateful_callbacks_equal = self._stateful_callbacks_equal(
            trainer_1.state.callbacks,
            trainer_2.state.callbacks,
        )
        if load_weights_only:
            # callback state should not have been loaded
            assert not stateful_callbacks_equal
            assert not metrics_equal
        else:
            assert stateful_callbacks_equal
            assert metrics_equal

    @pytest.mark.remote
    @device('cpu')
    @pytest.mark.parametrize('load_weights_only', [True, False])
    @pytest.mark.parametrize(
        'remote_checkpoint_uri, remote_checkpoint_name, continue_training_dur, final_checkpoint_name',
        [
            ['backwards_compatibility/trained_ckpt_cpu_ep2.pt', 'ep2.pt', '3ep', 'ep3.pt'],
        ],
    )
    @pytest.mark.filterwarnings('ignore:.*The checkpoint included CUDA RNG state.*')
    @pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                        reason='requires PyTorch 1.13 or higher')
    def test_load_remote_checkpoint(self, device, tmp_path: pathlib.Path, load_weights_only, remote_checkpoint_uri,
                                    remote_checkpoint_name, continue_training_dur, final_checkpoint_name, s3_bucket,
                                    s3_read_only_prefix):
        """
        This test checks if our checkpointing is backwards compatible.
        We should be able to load in a saved checkpoint and continue training.
        The checkpoint weight and metrics should match at load time
        and should be equivalent after training continues.
        Checkpoint saved using: Composer 0.13.5 with default dependencies.
        """
        trainer_1 = self.get_trainer(save_folder='first', device=device)
        trainer_1.fit()
        trainer_1.close()

        trainer_2 = self.get_trainer(
            max_duration=continue_training_dur,
            save_folder='second',
            load_path=f's3://{s3_bucket}/{s3_read_only_prefix}/{remote_checkpoint_uri}',
            load_weights_only=load_weights_only,
            load_strict_model_weights=load_weights_only,
            device=device,
        )

        # check weights loaded properly
        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

        # check metrics loaded
        metrics_equal = self._metrics_equal(trainer_1.state.train_metrics, trainer_2.state.train_metrics,
                                            trainer_1.state.eval_metrics, trainer_2.state.eval_metrics)

        if load_weights_only:
            assert not metrics_equal
            return

        assert metrics_equal

        # Continue training from old remote checkpoint
        trainer_2.fit()
        trainer_2.close()

        # Continue training from current local checkpoint
        trainer_3 = self.get_trainer(
            max_duration=continue_training_dur,
            save_folder='third',
            save_overwrite=True,
            load_path=os.path.join('first', remote_checkpoint_name),
            device=device,
        )
        trainer_3.fit()
        trainer_3.close()

        _assert_checkpoints_equivalent(os.path.join('third', final_checkpoint_name),
                                       os.path.join('second', final_checkpoint_name),
                                       rtol=1e-7,
                                       atol=1e-7)

    def _stateful_callbacks_equal(self, callbacks1, callbacks2):

        cb1 = next((cb for cb in callbacks1 if isinstance(cb, DummyStatefulCallback)))
        cb2 = next((cb for cb in callbacks2 if isinstance(cb, DummyStatefulCallback)))

        return cb1.random_value == cb2.random_value

    def test_load_weights_object_store(self, tmp_path):

        pytest.importorskip('libcloud')

        trainer_1 = self.get_trainer(
            save_folder='{run_name}/checkpoints',
            loggers=[self.get_logger(tmp_path)],
            run_name='electric-zebra',
        )
        trainer_1.fit()
        trainer_1.close()

        trainer_2 = self.get_trainer(
            loggers=[self.get_logger(tmp_path)],
            run_name='electric-zebra',
            load_path='electric-zebra/checkpoints/latest-rank0.pt',
            load_object_store=self.get_logger(tmp_path),
        )

        # check weights loaded properly
        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

        # check metrics loaded properly
        assert self._metrics_equal(
            trainer_1.state.train_metrics, trainer_2.state.train_metrics, trainer_1.state.eval_metrics,
            trainer_2.state.eval_metrics), 'Original metrics do not equal metrics from loaded checkpoint.'

    @pytest.mark.parametrize(
        'run_name,save_folder,save_overwrite,latest_filename',
        [
            [None, 'first', False, 'latest-rank{rank}.pt'],
            ['big-chungus', None, False, 'latest-rank{rank}.pt'],
            ['big-chungus', 'first', True, 'latest-rank{rank}.pt'],
            ['big-chungus', 'first', False, None],
        ],
    )
    def test_autoresume_fail(self, run_name, save_folder, save_overwrite, latest_filename):
        with pytest.raises(ValueError):
            self.get_trainer(
                latest_filename=latest_filename,
                save_overwrite=save_overwrite,
                save_folder=save_folder,
                run_name=run_name,
                autoresume=True,
            )

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

    def test_autoload_algorithm_old_checkpoint(self):
        trainer_1 = self.get_trainer(
            save_folder='first',
            algorithms=[NoOpModel()],
        )
        trainer_1.fit()
        trainer_1.close()

        trainer_2 = self.get_trainer(
            load_path=os.path.join('first', 'ep1.pt'),
            algorithms=[NoOpModel()],
        )
        trainer_2.fit(duration='1ba')

        # Monkeypatch algorithm to have different signature
        old_init, old_repr = NoOpModel.__init__, NoOpModel.__repr__
        NoOpModel.__init__ = lambda self, x: None  # type: ignore
        NoOpModel.__repr__ = lambda self: 'NoOpModel(3)'
        with pytest.warns(UserWarning, match='required_on_load algorithm.*'), pytest.raises(
                ValueError, match='loaded state dict contains a parameter group.*'):
            trainer_3 = self.get_trainer(load_path=os.path.join('first', 'ep1.pt'),)
            trainer_3.fit(duration='1ba')
        # Restore algorithm
        NoOpModel.__init__, NoOpModel.__repr__ = old_init, old_repr


class TestCheckpointResumption:

    def get_trainer(self,
                    model_init_device='cpu',
                    precision='fp32',
                    max_duration='2ep',
                    train_subset_num_batches=5,
                    **kwargs):
        model = SimpleModel()
        model.fc1.to(model_init_device)
        model.fc2.to(model_init_device)
        optimizer = torch.optim.Adam(model.parameters())

        train_dataset = RandomClassificationDataset(size=24)
        eval_dataset = RandomClassificationDataset(size=12)
        train_batch_size = 2

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                sampler=dist.get_sampler(train_dataset),
            ),
            eval_dataloader=DataLoader(
                dataset=eval_dataset,
                batch_size=2,
                sampler=dist.get_sampler(eval_dataset),
            ),
            device_train_microbatch_size=train_batch_size // 2,
            precision=precision,
            train_subset_num_batches=train_subset_num_batches,
            max_duration=max_duration,
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

        _assert_checkpoints_equivalent(
            save_folder / 'first' / final_checkpoint,
            save_folder / 'second' / final_checkpoint,
        )

    @pytest.mark.parametrize('world_size', [
        pytest.param(2, marks=pytest.mark.world_size(2)),
    ])
    @pytest.mark.parametrize('device', [
        pytest.param('gpu', marks=pytest.mark.gpu),
    ])
    @pytest.mark.parametrize(
        'use_orig_params,sync_module_states,model_1_init_device,model_2_init_device',
        [
            pytest.param(False, True, 'cpu', 'cpu'),  # success
            pytest.param(False, True, 'cpu', 'meta'),  # success
            pytest.param(True, True, 'cpu', 'cpu'),  # fail
            pytest.param(False, False, 'cpu', 'cpu'),  # fail
            pytest.param(False, True, 'meta', 'cpu'),  # fail
        ])
    @pytest.mark.filterwarnings('ignore:An unexpected prefix is detected. This case.*')
    @pytest.mark.filterwarnings(
        'ignore:``FullyShardedDataParallel.scatter_full_optim_state_dict``is being deprecated and is replaced by.*')
    def test_fsdp_monolith_resumption(
        self,
        device: str,
        world_size: int,
        use_orig_params: bool,
        sync_module_states: bool,
        model_1_init_device: str,
        model_2_init_device: str,
        tmp_path: pathlib.Path,
    ):
        save_interval = '1ba'
        save_filename = 'ba{batch}-rank{rank}.pt'
        resume_file = 'ba1-rank{rank}.pt'
        final_checkpoint = 'latest-rank{rank}.pt'
        fsdp_config = {
            'use_orig_params': use_orig_params,
            'sync_module_states': sync_module_states,
            'state_dict_type': 'full',
        }

        # All ranks use rank 0 folder
        tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
        save_folder = pathlib.Path(tmp_paths[0])

        trainer_1 = self.get_trainer(
            save_folder=os.path.join(save_folder, 'first'),
            save_filename=save_filename,
            save_interval=save_interval,
            eval_interval=save_interval,
            fsdp_config=fsdp_config,
            device=device,
            precision='amp_fp16',
            max_duration='1ep',
            train_subset_num_batches=2,
        )

        trainer_1.fit()
        trainer_1.close()

        self._assert_expected_num_checkpoints(
            save_folder=os.path.join(save_folder, 'first'),
            save_interval=save_interval,
            num_epochs=1,  # set in get_trainer()
            num_batches_per_epoch=2,  # set in get_trainer()
            is_deepspeed=False,
        )

        resume_file = os.path.join(save_folder, 'first', resume_file)
        model_init_device = [model_1_init_device, model_2_init_device][dist.get_global_rank()]
        fsdp_config['load_monolith_rank0_only'] = True

        success = use_orig_params == False and sync_module_states == True and model_1_init_device == 'cpu'
        with contextlib.nullcontext() if success else pytest.raises(ValueError):
            trainer_2 = self.get_trainer(
                model_init_device=model_init_device,
                save_folder=os.path.join(save_folder, 'second'),
                save_filename=save_filename,
                save_interval=save_interval,
                eval_interval=save_interval,
                fsdp_config=fsdp_config,
                device=device,
                precision='amp_fp16',
                max_duration='1ep',
                train_subset_num_batches=2,
                load_path=resume_file,  # <-- resume training from file
            )
            trainer_2.fit()
            trainer_2.close()

            _assert_checkpoints_equivalent(
                save_folder / 'first' / final_checkpoint,
                save_folder / 'second' / final_checkpoint,
            )

    @pytest.mark.parametrize('spin_dataloaders', [False, True])
    def test_spin_dataloaders(
        self,
        spin_dataloaders: bool,
        tmp_path: pathlib.Path,
    ):
        save_folder = tmp_path
        trainer_1 = self.get_trainer(
            save_folder=os.path.join(save_folder, 'first'),
            save_filename='ep{epoch}-rank{rank}.pt',
            save_interval='1ep',
        )

        trainer_1.fit()
        trainer_1.close()

        resume_file = os.path.join(save_folder, 'first', 'ep1-rank0.pt')
        trainer_2 = self.get_trainer(
            save_folder=os.path.join(save_folder, 'second'),
            save_filename='ep{epoch}-rank{rank}.pt',
            save_interval='1ep',
            load_path=resume_file,  # <-- resume training from file
            spin_dataloaders=spin_dataloaders,
        )
        trainer_2.fit()
        trainer_2.close()

        with contextlib.nullcontext() if spin_dataloaders else pytest.raises(AssertionError):
            _assert_checkpoints_equivalent(
                save_folder / 'first' / 'latest-rank{rank}.pt',
                save_folder / 'second' / 'latest-rank{rank}.pt',
            )

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
@pytest.mark.parametrize('num_keep', list(range(-1, 5)))
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
    num_keep,
    tmp_path: pathlib.Path,
):
    # all ranks use rank 0 folder
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = tmp_paths[0]

    deepseed_config = None
    if deepspeed_enabled:
        deepseed_config = {'zero_optimization': {'stage': zero_stage}}

    train_dataset = RandomImageDataset()

    trainer = Trainer(
        model=SimpleConvModel(),
        train_dataloader=DataLoader(
            dataset=train_dataset,
            sampler=dist.get_sampler(train_dataset),
        ),
        save_folder=str(save_folder),
        save_filename='checkpoint_{rank}_{batch}.pt',
        save_interval='1ba',
        max_duration='6ba',
        save_num_checkpoints_to_keep=num_keep,
        device=device,
        deepspeed_config=deepseed_config,
    )

    trainer.fit()

    dist.barrier()  # ensure all checkpoints rotated across ranks

    # deepspeed saves 1 file per rank
    total_checkpoints = 6
    num_keep = num_keep if num_keep >= 0 else total_checkpoints
    expected_num = num_keep if not deepspeed_enabled else num_keep * world_size

    files = glob(os.path.join(save_folder, 'checkpoint_*'))
    symlink_files = glob(os.path.join(save_folder, 'latest-rank*'))
    assert len(files) == expected_num
    assert len(symlink_files) == ((1 if not deepspeed_enabled else world_size) if num_keep != 0 else 0)

    dist.barrier()  # all ranks finish before cleaning up tmpdir
