# Copyright 2021 MosaicML. All Rights Reserved.

import os
import shutil
from typing import Optional, Tuple, Type

# import pathlib
import py
import pytest

from composer.algorithms import AlgorithmHparams
from composer.core.precision import Precision
from composer.datasets import DataLoaderHparams, DatasetHparams, SyntheticHparamsMixin
from composer.models import ModelHparams
from composer.optim import AdamHparams, ExponentialSchedulerHparams
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.trainer.trainer_hparams import TrainerHparams, algorithms_registry
from composer.utils import dist
from tests.trainer.test_checkpoint import assert_checkpoints_equivalent
from tests.utils.synthetic_utils import configure_dataset_for_synthetic, configure_model_for_synthetic

algo_hparams_overrides = {
    "swa": {
        "swa_start": "0.4dur",
        "swa_end": "0.8dur"
    },
    "cutmix": {
        "num_classes": None
    },
    "ghost_batchnorm": {
        "ghost_batch_size": 2
    },
    "stochastic_depth": {
        "target_layer_name": "ResNetBottleneck"
    },
    "selective_backprop": {
        "scale_factor": 1,
    }
}

xfail_list = [
    "blurpool",
    "squeeze_excite",
    "factorize",
    "layer_freezing",
    "sam",
    "stochastic_depth",  # timeout
    "swa"
]

skiplist = {
    'default': {
        'colout': 'Only for images',
        'cutmix': 'Only for images',
        'cutout': 'Only for images',
        'alibi': 'Not compatible with simple model.',
        'seq_length_warmup': 'Not compatible with simple model.',
        'randaugment': 'Requires PIL dataset to test.',
        'augmix': 'Required PIL dataset to test.',
        'stochastic_depth': 'Only applies to ResNets.',
        'no_op_model': 'Not compatible with this model.'
    },
    'resnet50_synthetic': {
        'alibi': 'Not compatible with vision model.',
        'seq_length_warmup': 'Not compatible with vision model.',
        'randaugment': 'Requires PIL dataset to test.',
        'augmix': 'Required PIL dataset to test.',
        'no_op_model': 'Not compatible with this model.'
    },
    'gpt2_52m': {
        'seq_length_warmup': "issue with dataset",
        'blurpool': 'Only for CNNs',
        'ghost_batchnorm': 'No BatchNorm',
        'label_smoothing': 'Not applicable',
        'selective_backprop': 'Not applicable',
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
}


@pytest.mark.timeout(180)
@pytest.mark.parametrize("device_hparams", [
    pytest.param(CPUDeviceHparams(), id="cpu"),
    pytest.param(GPUDeviceHparams(), id="gpu", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize(
    "seed,save_interval,save_filename,resume_file,final_checkpoint",
    [
        [None, "1ep", "ep{epoch}-rank{rank}", "ep2-rank{rank}", "latest-rank{rank}"
        ],  # test randomized seed saving and symlinking
        [42, "1ep", "ep{epoch}-rank{rank}", "ep3-rank{rank}", "ep5-rank{rank}"],  # test save at epoch end
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [pytest.param("default"), pytest.param("resnet50_synthetic", marks=pytest.mark.daily)])
@pytest.mark.parametrize("algorithm", algorithms_registry.items(), ids=algorithms_registry.keys())
def test_algorithm_resumption(
    device_hparams: DeviceHparams,
    save_interval: str,
    save_filename: str,
    resume_file: str,
    final_checkpoint: str,
    seed: Optional[int],
    model_name: str,
    algorithm: Tuple[str, Type[AlgorithmHparams]],
    dummy_model_hparams: ModelHparams,
    dummy_train_dataset_hparams: DatasetHparams,
    dummy_val_dataset_hparams: DatasetHparams,
    tmpdir: py.path.local,
):
    """strategy:
    - train five epochs. capture checkpoints after `checkpoint_interval` and ep5.
    - create a new trainer from the `checkpoint_interval` checkpoint, and train until end. checkpoint again.
    - assert that the checkpoint from the new trainer at the end is the same as the checkpoint from the first trainer at the end.
    """

    algo_name, algo_hparams = algorithm

    if algo_name in xfail_list:
        pytest.xfail(f"{algo_name} known to fail resumption test")

    if algo_name in skiplist[model_name].keys():
        pytest.skip(skiplist[model_name][algo_name])

    max_epochs = "5ep"
    subset_num_batches = 5

    trainer_hparams = TrainerHparams(algorithms=[],
                                     seed=seed,
                                     optimizer=AdamHparams(),
                                     schedulers=[ExponentialSchedulerHparams(gamma=0.9)],
                                     precision=Precision.FP32,
                                     max_duration=max_epochs,
                                     train_batch_size=8,
                                     eval_batch_size=8,
                                     dataloader=DataLoaderHparams(
                                         num_workers=0,
                                         prefetch_factor=2,
                                         persistent_workers=False,
                                         pin_memory=False,
                                         timeout=0.0,
                                     ),
                                     model=dummy_model_hparams,
                                     val_dataset=dummy_val_dataset_hparams,
                                     train_dataset=dummy_train_dataset_hparams,
                                     deterministic_mode=True,
                                     loggers=[],
                                     grad_accum=2,
                                     train_subset_num_batches=subset_num_batches,
                                     eval_subset_num_batches=subset_num_batches,
                                     save_filename=save_filename,
                                     device=device_hparams)

    if model_name != "default":
        if not isinstance(device_hparams, GPUDeviceHparams):
            pytest.skip("Real models require a GPU -- otherwise they take too long")
        model_hparams = TrainerHparams.load(model_name)
        trainer_hparams.train_dataset = model_hparams.train_dataset
        trainer_hparams.val_dataset = model_hparams.val_dataset
        trainer_hparams.model = model_hparams.model
        trainer_hparams.optimizer = model_hparams.optimizer
        trainer_hparams.schedulers = model_hparams.schedulers

    if not isinstance(trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")

    if not isinstance(trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")

    if algo_name == "cutmix":
        algo_hparams_overrides["cutmix"]["num_classes"] = trainer_hparams.model.num_classes

    if algo_name in algo_hparams_overrides:
        algo_object = algo_hparams(**algo_hparams_overrides[algo_name])
    else:
        algo_object = algo_hparams()

    configure_model_for_synthetic(trainer_hparams.model)
    configure_dataset_for_synthetic(trainer_hparams.train_dataset, trainer_hparams.model)
    configure_dataset_for_synthetic(trainer_hparams.val_dataset, trainer_hparams.model)

    trainer_hparams.algorithms = [algo_object]
    trainer_hparams.train_dataset.use_synthetic = True
    trainer_hparams.train_dataset.synthetic_num_unique_samples = 500
    trainer_hparams.train_dataset.shuffle = False
    trainer_hparams.val_dataset.use_synthetic = True
    trainer_hparams.val_dataset.shuffle = False

    checkpoint_a_folder = str(tmpdir / "first")
    trainer_hparams.save_folder = checkpoint_a_folder
    trainer_hparams.save_interval = save_interval

    trainer_hparams.validate_every_n_batches = 1 if resume_file.startswith("ba") else 0
    trainer_hparams.validate_every_n_epochs = 1 if resume_file.startswith("ep") else 0

    first_trainer = trainer_hparams.initialize_object()
    first_trainer.fit()

    rank_to_checkpoint_a_folder = dist.all_gather_object(os.path.abspath(checkpoint_a_folder))

    checkpoint_to_resume_filepath = os.path.join(rank_to_checkpoint_a_folder[0], resume_file)
    first_trainer_final_checkpoint_filepath = os.path.join(rank_to_checkpoint_a_folder[0], final_checkpoint)

    # Move the resume and final file to the rank 0 folder
    try:
        rank_checkpoint_filepath = os.path.join(checkpoint_a_folder, resume_file.format(rank=dist.get_global_rank()))
        shutil.copy2(rank_checkpoint_filepath,
                     checkpoint_to_resume_filepath.format(rank=dist.get_global_rank()),
                     follow_symlinks=True)
    except (shutil.SameFileError, FileNotFoundError):
        pass

    try:
        rank_checkpoint_filepath = os.path.join(checkpoint_a_folder,
                                                final_checkpoint.format(rank=dist.get_global_rank()))
        shutil.copy2(rank_checkpoint_filepath,
                     first_trainer_final_checkpoint_filepath.format(rank=dist.get_global_rank()),
                     follow_symlinks=True)
    except (shutil.SameFileError, FileNotFoundError):
        pass

    second_trainer_hparams = TrainerHparams.create(data=trainer_hparams.to_dict(), cli_args=False)
    checkpoint_b_folder = os.path.join(rank_to_checkpoint_a_folder[0], "second")

    second_trainer_hparams.save_folder = checkpoint_b_folder
    second_trainer_hparams.load_path = checkpoint_to_resume_filepath
    second_trainer_hparams.load_weights_only = False
    second_trainer_hparams.load_strict_model_weights = False

    second_trainer = second_trainer_hparams.initialize_object()
    second_trainer.fit()
    second_trainer_final_checkpoint_filepath = os.path.join(checkpoint_b_folder, final_checkpoint)

    assert_checkpoints_equivalent(hparams_a=trainer_hparams,
                                  checkpoint_file_a=first_trainer_final_checkpoint_filepath,
                                  hparams_b=second_trainer_hparams,
                                  checkpoint_file_b=second_trainer_final_checkpoint_filepath,
                                  state_attrs_to_skip=[
                                      '_optimizers',
                                      '_schedulers',
                                      '_algorithms',
                                      '_callbacks',
                                  ])

    # Clean up passed checkpoints
    tmpdir.remove()
