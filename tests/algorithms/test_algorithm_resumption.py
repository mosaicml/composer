# Copyright 2021 MosaicML. All Rights Reserved.

import os
import textwrap
from typing import Optional, Type, Tuple

import pytest

from composer.algorithms import AlgorithmHparams
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.trainer.trainer_hparams import algorithms_registry
from composer.core.precision import Precision
from composer.core.time import Time, TimeUnit
from composer.datasets import SyntheticHparamsMixin
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import run_directory
from composer.utils.checkpoint import _is_archive
from tests.fixtures.dummy_fixtures import dummy_num_classes
from tests.trainer.test_checkpoint import assert_checkpoints_equivalent


algo_hparams_overrides = {
    "swa": {
        "swa_start": "0.4dur",
        "swa_end": "0.8dur"
    },
    "cutmix": {
        "num_classes": dummy_num_classes
    },
    "mixup": {
        "num_classes": dummy_num_classes
    }    
}

simple_net_skiplist = {
            'colout': 'Only for images',
            'cutmix': 'Only for images',
            'cutout': 'Only for images',            
            'mixup': 'Only defined for images',
            'alibi': 'Not compatible with simple model.',
            'seq_length_warmup': 'Not compatible with simple model.',
            'randaugment': 'Requires PIL dataset to test.',
            'augmix': 'Required PIL dataset to test.',
            'stochastic_depth': 'Only applies to ResNets.',
            'no_op_model': 'Not compatible with this model.'
        }

resnet_skiplist = {
            'alibi': 'Not compatible with simple model.',
            'seq_length_warmup': 'Not compatible with simple model.',
            'randaugment': 'Requires PIL dataset to test.',
            'augmix': 'Required PIL dataset to test.',
            'no_op_model': 'Not compatible with this model.'
        }

lm_skiplist = {
            'blurpool': 'Only for CNNs',
            'channels_last': 'Only for CNNs',
            'colout': 'Only for images',
            'cutmix': 'Only for images',
            'cutout': 'Only for images',
            'squeeze_excite': 'Not defined for language model transformers',
            'mixup': 'Only defined for images',
            'randaugment': 'Requires PIL dataset to test.',
            'augmix': 'Required PIL dataset to test.',
            'progressive_resizing': 'Only defined for images',
            'stochastic_depth': 'Only applies to ResNets.',
            'no_op_model': 'Not compatible with this model.'
        }


@pytest.mark.timeout(180)
@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize("device_hparams,deepspeed_enabled,zero_stage", [
    pytest.param(CPUDeviceHparams(), False, None, id="cpu-ddp"),
    pytest.param(GPUDeviceHparams(), False, None, id="gpu-ddp", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), True, 0, id="deepspeed-zero0", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), True, 1, id="deepspeed-zero1", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), True, 2, id="deepspeed-zero2", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize(
    "seed,save_interval,save_name_format,resume_file,final_checkpoint",
    [
        [None, "1ep", "ep{epoch}", "ep3", "latest-rank{rank}"],  # test randomized seed saving and symlinking
        [42, "1ep", "ep{epoch}", "ep3", "ep5"],  # test save at epoch end
        [42, "1ep", "ep{epoch}.tgz", "ep3.tgz", "ep5.tgz"],  # test tarball with compression
        [42, "2ba", "ba{batch}", "ba12", "ba25"],  # test save batch in partial epoch
        [42, "1ba", "ba{batch}", "ba15", "ba25"],  # test save final batch in epoch
        [42, "2ba", "ba{batch}", "ba16", "ba25"],  # test save batch after complete epoch
    ],
)
@pytest.mark.parametrize("model_name", [None, "resnet50_synthetic", "gpt2_52m"])
@pytest.mark.parametrize("algorithm", algorithms_registry.items(), ids=algorithms_registry.keys())
def test_algorithm_resumption(
    device_hparams: DeviceHparams,
    world_size: int,
    deepspeed_enabled: bool,
    zero_stage: Optional[int],
    composer_trainer_hparams: TrainerHparams,
    save_interval: str,
    save_name_format: str,
    resume_file: str,
    final_checkpoint: str,
    seed: Optional[int],
    model_name: Optional[str],
    algorithm: Tuple[str, Type[AlgorithmHparams]],
):
    """strategy:
    - train five epochs. capture checkpoints after `checkpoint_interval` and ep5.
    - create a new trainer from the `checkpoint_interval` checkpoint, and train until end. checkpoint again.
    - assert that the checkpoint from the new trainer at the end is the same as the checkpoint from the first trainer at the end.
    """
    del world_size  # unused. Read via env variable

    algo_name, algo_hparams = algorithm

    if not isinstance(device_hparams, GPUDeviceHparams) and deepspeed_enabled:
        pytest.skip("DeepSpeed tests must be ran on GPU")

    if deepspeed_enabled:
        if not _is_archive(resume_file):
            resume_file += ".tar"
        if not _is_archive(final_checkpoint):
            final_checkpoint += ".tar"

    composer_trainer_hparams.schedulers = []
    composer_trainer_hparams.max_duration = '5ep'
    if model_name in ["resnet50_synthetic", "gpt2_52m"]:
        if not isinstance(device_hparams, GPUDeviceHparams):
            pytest.skip("Real models require a GPU -- otherwise they take too long")
        model_hparams = TrainerHparams.load(model_name)
        composer_trainer_hparams.train_dataset = model_hparams.train_dataset
        composer_trainer_hparams.val_dataset = model_hparams.val_dataset
        composer_trainer_hparams.model = model_hparams.model
        composer_trainer_hparams.optimizer = model_hparams.optimizer
        composer_trainer_hparams.schedulers = model_hparams.schedulers
    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")
        return

    if model_name is None and algo_name in simple_net_skiplist:
        pytest.skip(simple_net_skiplist[algo_name])

    if model_name == 'resnet50_synthetic' and algo_name in resnet_skiplist:
        pytest.skip(resnet_skiplist[algo_name])
    
    if model_name == 'gpt2_52m' and algo_name in lm_skiplist:
        pytest.skip(lm_skiplist[algo_name])

    if algo_name in algo_hparams_overrides:
        algo_object = algo_hparams(**algo_hparams_overrides[algo_name])
    else:
        algo_object = algo_hparams()

    composer_trainer_hparams.algorithms = [algo_object]
    composer_trainer_hparams.save_name_format = save_name_format
    composer_trainer_hparams.train_dataset.use_synthetic = True
    composer_trainer_hparams.train_dataset.synthetic_num_unique_samples = 500
    composer_trainer_hparams.train_dataset.shuffle = False
    composer_trainer_hparams.val_dataset.use_synthetic = True
    composer_trainer_hparams.val_dataset.shuffle = False
    composer_trainer_hparams.grad_accum = 2
    composer_trainer_hparams.loggers = []
    composer_trainer_hparams.train_batch_size = 10
    composer_trainer_hparams.eval_batch_size = 10
    num_epochs = 5
    composer_trainer_hparams.max_duration = f"{num_epochs}ep"
    composer_trainer_hparams.precision = Precision.FP32
    composer_trainer_hparams.train_subset_num_batches = 5
    composer_trainer_hparams.eval_subset_num_batches = 5
    composer_trainer_hparams.device = device_hparams
    if deepspeed_enabled:
        assert zero_stage is not None
        if zero_stage > 0:
            composer_trainer_hparams.deterministic_mode = False
            if model_name is not None:
                pytest.skip(
                    textwrap.dedent(f"""\
                        Skipping test since deterministic mode is required for
                        non-trivial models, but deterministic mode isn't compatible with deepspeed
                        zero stage {zero_stage}"""))
        composer_trainer_hparams.deepspeed = {"zero_optimization": {"stage": zero_stage}}

    checkpoint_a_folder = "first"
    composer_trainer_hparams.save_folder = checkpoint_a_folder
    composer_trainer_hparams.save_interval = save_interval
    composer_trainer_hparams.seed = seed

    composer_trainer_hparams.validate_every_n_batches = 1 if resume_file.startswith("ba") else 0
    composer_trainer_hparams.validate_every_n_epochs = 1 if resume_file.startswith("ep") else 0
    first_trainer = composer_trainer_hparams.initialize_object()
    first_trainer.fit()
    save_interval_time = Time.from_timestring(save_interval)
    if save_interval_time.unit == TimeUnit.EPOCH:
        expected_num_checkpoints = ((num_epochs - 1) // save_interval_time.value) + 1
    else:
        expected_num_checkpoints = (
            (composer_trainer_hparams.train_subset_num_batches * num_epochs - 1) // save_interval_time.value) + 1
    checkpoint_saver = None
    for callback in first_trainer.state.callbacks:
        if isinstance(callback, CheckpointSaver):
            checkpoint_saver = callback
    assert checkpoint_saver is not None
    assert len(checkpoint_saver.saved_checkpoints) == expected_num_checkpoints
    checkpoint_a_file_path = os.path.join(checkpoint_a_folder, resume_file)
    checkpoint_b_file_path = os.path.join(run_directory.get_node_run_directory(), "rank_{rank}", checkpoint_a_folder,
                                          final_checkpoint)

    second_trainer_hparams = TrainerHparams.create(data=composer_trainer_hparams.to_dict(), cli_args=False)
    checkpoint_b_folder = "second"

    second_trainer_hparams.save_folder = checkpoint_b_folder
    second_trainer_filepath = os.path.join(run_directory.get_node_run_directory(), "rank_{rank}",
                                           checkpoint_a_file_path)
    second_trainer_hparams.load_path_format = second_trainer_filepath
    second_trainer_hparams.load_weights_only = False
    second_trainer_hparams.load_strict_model_weights = False

    second_trainer = second_trainer_hparams.initialize_object()
    second_trainer.fit()
    checkpoint_c_file_path = os.path.join(run_directory.get_node_run_directory(), "rank_{rank}", checkpoint_b_folder,
                                          final_checkpoint)

    assert_checkpoints_equivalent(
        hparams_a=composer_trainer_hparams,
        checkpoint_file_a=checkpoint_b_file_path,
        hparams_b=second_trainer_hparams,
        checkpoint_file_b=checkpoint_c_file_path,
    )
