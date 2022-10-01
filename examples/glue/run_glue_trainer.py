# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Specifically designed for the NLP use case, allowing pre-training and fine-tuning on
downstream tasks to be handled within one script. This script requires that the
run_composer_trainer.py script lies in the parent folder to this one.

Example that pretrains a BERT::
    >>> python examples/glue/run_glue_trainer.py
    -f examples/glue/glue_example.yaml
    --training_scheme pretrain

Example that pretrains and finetunes a BERT::
    >>> python examples/glue/run_glue_trainer.py
    -f examples/glue/glue_example.yaml
    --training_scheme all

Example that finetunes a pretrained BERT::

    >>> python examples/glue/run_glue_trainer.py
    -f examples/glue/glue_example.yaml
    --training_scheme finetune

To see all the possible options for a specific parameter usage,
try ``python examples/glue/run_glue_trainer.py <PARAMETER_NAME> --help``
like in the following::

    >>> python examples/glue/run_glue_trainer.py
    finetune_hparams --help
"""
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yahp as hp
import yaml
from nlp_trainer_hparams import GLUETrainerHparams, NLPTrainerHparams
from tabulate import tabulate

from composer.core.data_spec import DataSpec
from composer.core.time import Time, Timestamp, TimeUnit
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils.file_helpers import format_name_with_dist_and_time
from composer.utils.misc import get_free_tcp_port, warning_on_one_line

__all__ = ['GlueMetricsState', 'GlueState']


class GlueMetricsState:
    """Class mapping all GLUE tasks to their respective average metric values.

    Args:
        task_names (List[str]): the names of the GLUE tasks stored in the data struct
    """

    def __init__(self, task_names: List[str]) -> None:
        self.task_to_avg_metric = {}

        for task in task_names:
            self.task_to_avg_metric[task] = None


@dataclass
class GlueState:
    """Class storing all GLUE metrics per checkpoint collected during a finetuning job spawned by the NLP entrypoint.

    This class maps checkpoint names to GlueMetricsState instances which map tasks to their respective average
    metric values.

    Args:
        task_names (List[str]): See :class:`.GLUEMetricsState`
        ckpt_to_tasks (Dict[str, GLUEMetricsState]): dictionary mapping checkpoint names to :class:`.GLUEMetricsState`
    """
    task_names: List[str]
    ckpt_to_tasks: Dict[str, GlueMetricsState]


def log_metrics(metric: Dict[str, Dict], task: str, ckpt_filename: str, glue_metrics: GlueState) -> None:
    """Callback function for metric collection.

    Args:
        metric (Dict): Metrics returned from ``train_finetune()`` for a given GLUE task.
        task (str): Task to log metrics under.
        ckpt_filename (str): Checkpoint to log metrics under.
        glue_metrics (GlueState): GlueState object storing all the glue metrics for the entrypoint's current run.
    """
    if ckpt_filename not in glue_metrics.ckpt_to_tasks.keys():
        glue_metrics.ckpt_to_tasks[ckpt_filename] = GlueMetricsState(glue_metrics.task_names)

    formatted_task = task.lower()
    for _, evaluator_metrics in metric.items():  # handle case where mnli has glue_mnli and glue_mnli_mismatched
        for _, metric_val in evaluator_metrics.items():  # handle case where an evaluator has multiple metrics
            tasks = glue_metrics.ckpt_to_tasks[ckpt_filename]
            task_metric = tasks.task_to_avg_metric[formatted_task]
            if not task_metric:
                tasks.task_to_avg_metric[formatted_task] = []
            tasks.task_to_avg_metric[formatted_task].append(metric_val)


def print_metrics(glue_metrics: GlueState) -> None:
    """Consolidate and prettify metrics."""
    tasks = glue_metrics.task_names
    large_tasks = ['mnli', 'qnli', 'qqp', 'sst-2']
    assert all(task in glue_metrics.task_names for task in large_tasks)
    # init table headers
    headers = ['Checkpoint']
    headers.extend([f'{task.upper()}' for task in sorted(tasks)])
    headers.extend(['GLUE-Large', 'GLUE-Avg'])
    tb = [headers]

    empty_str = '  --  '
    # fill table
    for ckpt in glue_metrics.ckpt_to_tasks.keys():
        output_line = [ckpt]
        glue_all = 0
        glue_large = 0
        count_all = 0
        count_large = 0

        # Per task score
        for task in sorted(glue_metrics.task_names):
            task_metric = glue_metrics.ckpt_to_tasks[ckpt].task_to_avg_metric[task]
            if not task_metric:  # Empty if this task wasn't run for some reason
                output_line.append(empty_str)
                continue
            assert isinstance(task_metric, list)
            logged_metric = float(np.nanmean(task_metric))
            output_line.append('{:.4f}'.format(logged_metric))
            glue_all += logged_metric
            count_all += 1
            if task in large_tasks:
                glue_large += logged_metric
                count_large += 1

        # GLUE Large and GLUE All
        if count_large > 0:
            output_line.append('{:.4f}'.format(glue_large / count_large))
        else:
            output_line.append(empty_str)
        if count_all > 0:
            output_line.append('{:.4f}'.format(glue_all / count_all))
        else:
            output_line.append(empty_str)
        tb.append(output_line)

    print(tabulate(tb, headers='firstrow'))


def merge_hparams(hparams: TrainerHparams, override_hparams: GLUETrainerHparams) -> TrainerHparams:
    """Overrides the atttributes of the hparams instance with those of the provided override_hparams."""
    hparams.algorithms = override_hparams.algorithms if override_hparams.algorithms else hparams.algorithms
    hparams.load_ignore_keys = override_hparams.load_ignore_keys if override_hparams.load_ignore_keys else hparams.load_ignore_keys
    hparams.load_path = override_hparams.load_path if override_hparams.load_path else hparams.load_path
    hparams.load_object_store = override_hparams.load_object_store if override_hparams.load_object_store else hparams.load_object_store
    hparams.load_logger_destination = override_hparams.load_logger_destination if override_hparams.load_logger_destination else hparams.load_logger_destination
    hparams.loggers = override_hparams.loggers if override_hparams.loggers else hparams.loggers
    hparams.model = override_hparams.model if override_hparams.model else hparams.model
    hparams.run_name = override_hparams.run_name if override_hparams.run_name else hparams.run_name
    hparams.save_folder = override_hparams.save_folder if override_hparams.save_folder else hparams.save_folder

    return hparams


def _setup_gpu_queue(num_gpus: int, manager: SyncManager):
    """Returns a queue with [0, 1, .. num_gpus]."""
    gpu_queue = manager.Queue(num_gpus)
    for gpu_id in range(num_gpus):
        gpu_queue.put(gpu_id)
    return gpu_queue


def spawn_finetuning_jobs(
    task_to_save_ckpt: Dict[str, bool],
    ckpt_load_paths: List[str],
    save_folder: str,
    base_yaml_file: str,
    save_locally: bool,
    load_locally: bool,
    parent_ckpts: Optional[List[str]] = None,
    load_ignore_keys: Optional[List[str]] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Set up CUDA environment and process pool for given finetuning jobs and wait for them to complete."""
    if parent_ckpts:
        assert len(parent_ckpts) == len(ckpt_load_paths), 'Must supply one parent_ckpt per ckpt_load_path'
    else:
        parent_ckpts = ckpt_load_paths  # By default, the "parent checkpoint" is logged simply as the checkpoint

    # To reduce noisiness with these tasks, expand the evaluation to multiple fine-tuning seeds per pre-train checkpoint, if desired
    fthp = NLPTrainerHparams.create(cli_args=False, f=base_yaml_file).finetune_hparams
    assert fthp is not None
    seed_overrides = fthp.seed_overrides
    if seed_overrides is None:
        seed_overrides = {}
    else:
        seed_overrides = {k.lower(): v for k, v in seed_overrides.items()}

    num_gpus = torch.cuda.device_count()
    free_port = get_free_tcp_port()

    with mp.Manager() as manager:

        # workers get gpu ids from this queue
        # to set the GPU to run on
        gpu_queue = _setup_gpu_queue(num_gpus, manager)

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_gpus, maxtasksperchild=1) as pool:
            results = []
            rank = 0
            # Fine-tune from pre-trained checkpoint(s)
            ckpt_parent_pairs = zip(ckpt_load_paths, parent_ckpts)
            for parent_idx, ckpt_parent_pair in enumerate(ckpt_parent_pairs):
                ckpt_load_path, parent_ckpt = ckpt_parent_pair
                # `ckpt_load_path` provides the path to the checkpoint from which we load the starting weights used when fine-tuning
                # `parent_ckpt` keeps track of the original pre-training checkpoint, for tasks with multiple fine-tuning stages (e.g., RTE)
                # `parent_idx` is used for bookkeeping, so `parent_ckpt` can be internally recovered from the path used to save fine-tune checkpoints
                for task, save_ckpt in task_to_save_ckpt.items():
                    # Run 1 or more fine-tune trainers from this checkpoint, using a different seed override for each
                    for seed in seed_overrides.get(task, [None]):
                        result = pool.apply_async(
                            train_finetune,
                            args=(gpu_queue, base_yaml_file, task, save_ckpt, ckpt_load_path, parent_ckpt, parent_idx,
                                  save_folder, save_locally, load_locally, free_port + rank, load_ignore_keys, seed),
                        )
                        results.append(result)
                        rank += 1

            pool.close()
            pool.join()

    finished_results = [result.get() for result in results]
    return finished_results


def train_finetune(
        gpu_queue: mp.Queue,
        base_yaml_file: str,
        task: str,
        save_ckpt: bool,
        load_path: str,
        parent_ckpt: str,
        parent_idx: int,
        save_folder: str,
        save_locally: bool,
        load_locally: bool,
        master_port: int,
        load_ignore_keys: Optional[List[str]] = None,
        seed_override: Optional[int] = None,  # Option to manually set the seed to this value
) -> Tuple[str, str, Dict[str, Any]]:
    """Run single instance of a finetuning job on given task."""
    os.environ['MASTER_PORT'] = f'{master_port}'  # set unique master port for each spawn

    gpu_id = gpu_queue.get() if gpu_queue else 0
    torch.cuda.set_device(gpu_id)

    finetune_hparams = NLPTrainerHparams.create(cli_args=False, f=base_yaml_file).finetune_hparams
    task_hparams = TrainerHparams.create(cli_args=False, f=f'./composer/yamls/models/glue/{task}.yaml')
    # turn off multiple workers on the dataloader for multiprocessing
    # since all of the data is cached locally for GLUE, multiple workers don't hurt us.
    task_hparams.dataloader.num_workers = 0
    task_hparams.dataloader.persistent_workers = False

    if finetune_hparams:
        ft_hparams = merge_hparams(task_hparams, finetune_hparams)
    else:
        ft_hparams = task_hparams

    ft_hparams.load_path = load_path
    ft_hparams.device = DeviceGPU(
        torch.cuda.current_device())  # set device manually to force finetuning to happen on one GPU
    ft_hparams.log_to_console = False
    ft_hparams.progress_bar = False
    ft_hparams.save_overwrite = True

    if load_ignore_keys is not None:
        if ft_hparams.load_ignore_keys:
            ft_hparams.load_ignore_keys.extend(load_ignore_keys)
        else:
            ft_hparams.load_ignore_keys = load_ignore_keys

    if seed_override is not None:
        assert seed_override > 0
        ft_hparams.seed = seed_override

    # add finetune-specific tags to wandb if logger exists
    if ft_hparams.loggers:
        for logger in ft_hparams.loggers:
            if isinstance(logger, WandBLogger):
                if 'tags' not in logger._init_kwargs.keys():
                    logger._init_kwargs['tags'] = []
                logger._init_kwargs['tags'].append(task)

    if load_locally:
        ft_hparams.load_object_store = None

    # saving single checkpoint at the end of training the task
    if save_ckpt:
        # add task specific artifact logging information
        ft_hparams.save_folder = f'{save_folder}/{task}-{parent_idx:03d}'
        save_artifact_name = f'{save_folder}/{task}-{parent_idx:03d}/ep{{epoch}}-ba{{batch}}-rank{{rank}}'  # ignored if not uploading
        save_latest_artifact_name = f'{save_folder}/{task}-{parent_idx:03d}/latest-rank{{rank}}'
        ft_hparams.save_artifact_name = save_artifact_name
        ft_hparams.save_latest_artifact_name = save_latest_artifact_name

        if save_locally:
            if not os.path.exists(ft_hparams.save_folder):
                os.makedirs(ft_hparams.save_folder)

    else:
        # Disable saving
        ft_hparams.save_folder = None

    print(
        f'\n --------\n SPAWNING TASK {task.upper()}\n DEVICE: {torch.cuda.current_device()}\n CKPT: {parent_ckpt}\n --------'
    )

    try:
        trainer = ft_hparams.initialize_object()

        # if using wandb, store the config and other information inside the wandb run
        try:
            import wandb
        except ImportError:
            pass
        else:
            if wandb.run is not None:
                wandb.config.update(ft_hparams.to_dict())
                wandb.config.update({'pretrained_ckpt': parent_ckpt, 'task': task, 'pretrained_idx': parent_idx})

        trainer.fit()
        print(f'\nFINISHED TRAINING TASK {task.upper()}\n')

        # cpu_metrics = DeviceCPU().batch_to_device(trainer.state.eval_metrics)  # <-- Updating API call
        collected_metrics: Dict[str, Dict[str, Any]] = {}
        for eval_name, metrics in trainer.state.eval_metrics.items():
            collected_metrics[eval_name] = {name: metric.compute().cpu().numpy() for name, metric in metrics.items()}
        trainer.close()
    finally:
        print(f'Releasing GPU {gpu_id}')
        gpu_queue.put(gpu_id)

    return task, parent_ckpt, collected_metrics


def get_args() -> str:
    """Get NLPTrainerHparams arguments from CLI."""
    parser = hp.get_argparse(NLPTrainerHparams)
    args, _ = parser.parse_known_args()
    return args.file


def validate_args(hp: NLPTrainerHparams) -> None:
    """Validate CLI args as well as finetune-specific parameters."""
    if hp.training_scheme not in ('finetune', 'pretrain', 'all'):
        raise ValueError('training_scheme must be one of "finetune", "pretrain," or "all"')

    if hp.training_scheme != 'finetune' and not hp.pretrain_hparams:
        raise ValueError('pretrain_hparams must be specified if pretraining a model')

    elif hp.training_scheme == 'finetune' and ((not hp.finetune_hparams) or
                                               (hp.finetune_hparams and not hp.finetune_hparams.finetune_ckpts)):
        raise ValueError('load_path to checkpoints must be specified if finetuning a model')

    elif hp.training_scheme == 'pretrain' and hp.finetune_hparams:
        warnings.warn('finetune_hparams specified. These values will be ignored during pretraining.')

    elif hp.training_scheme == 'all' and hp.finetune_hparams is None:
        warnings.warn('No shared finetune_hparams specified. All finetune tasks will use their default configurations.')

    elif hp.training_scheme == 'all' and hp.finetune_hparams and hp.finetune_hparams.finetune_ckpts:
        warnings.warn('finetune_ckpts specified in finetune_hparams. This value will be overriden during finetuning.')

    if hp.finetune_hparams is not None:
        seed_overrides = hp.finetune_hparams.seed_overrides
        if seed_overrides is not None:
            assert isinstance(seed_overrides, dict)
            for task, seeds in seed_overrides.items():
                if task.lower() not in ['mnli', 'qnli', 'qqp', 'sst-2', 'cola', 'rte', 'mrpc', 'stsb']:
                    raise KeyError(f'Key "{task}" in finetune_hparams.seed_overrides is not a GLUE task.')
                if not isinstance(seeds, (tuple, list)):
                    raise TypeError(f'Seed overrides for task "{task}" must be a tuple or list of positive integers')
                for seed in seeds:
                    if (not isinstance(seed, int)) or seed <= 0:
                        raise TypeError(
                            f'Seed overrides for task "{task}" must be a tuple or list of positive integers')


def get_finetune_hparams() -> Tuple[GLUETrainerHparams, str, bool, bool]:
    """Extract finetune-specific hparams from the provided file and add entrypoint specific args to it."""
    hp = NLPTrainerHparams.create()
    validate_args(hp)

    training_scheme = hp.training_scheme

    save_locally = True
    load_locally = True
    hparams = GLUETrainerHparams(model=None)
    if training_scheme in ('finetune', 'all'):
        if hp.finetune_hparams:
            hparams = hp.finetune_hparams
            if hparams.finetune_ckpts:
                load_locally = False
            if hparams.load_object_store:
                load_locally = False
            if hparams.loggers:
                for l in hparams.loggers:
                    if isinstance(l, ObjectStoreLogger):
                        save_locally = False
                    if isinstance(l, WandBLogger) and l._log_artifacts:
                        save_locally = False

    return hparams, training_scheme, save_locally, load_locally


def get_ckpt_names(hp: TrainerHparams, run_name: str, dataloader_len: int) -> List[str]:
    """Extract list of checkpoints that will be saved by the given configuration."""
    ckpt_names = []
    assert hp.save_interval is not None
    assert hp.max_duration is not None
    interval = Time.from_timestring(str(hp.save_interval))
    duration = Time.from_timestring(str(hp.max_duration))

    ep = 0
    ba = 0
    loop = True
    save = False
    save_last_batch = False
    while loop:
        if save:
            time = Timestamp(epoch=ep, batch=ba)
            formatted_ckpt_name = format_name_with_dist_and_time(hp.save_artifact_name, run_name, time)
            ckpt_names.append(formatted_ckpt_name)
            save = False

        ba += interval.value
        if interval.unit == TimeUnit.BATCH:
            save = True
        if ba >= dataloader_len:  # batches per epoch
            ep += 1
            if interval.unit == TimeUnit.EPOCH:
                save = True

        if duration.unit == TimeUnit.BATCH:
            if ba >= duration.value:
                loop = False
                save_last_batch = True
                if ba > duration.value:
                    ba = duration.value
        elif duration.unit == TimeUnit.EPOCH:
            if ep >= duration.value:
                loop = False
        elif duration.unit == TimeUnit.SAMPLE:
            if ba * hp.train_batch_size >= duration.value:
                loop = False
                save_last_batch = True
                if ba * hp.train_batch_size > duration.value:
                    ba = duration.value // hp.train_batch_size

    # save very last batch if incrementing batches passed it
    if save_last_batch:
        time = Timestamp(epoch=ep, batch=ba)
        formatted_ckpt_name = format_name_with_dist_and_time(hp.save_artifact_name, run_name, time)
        ckpt_names.append(formatted_ckpt_name)

    return ckpt_names


def run_pretrainer(training_scheme: str, file: str, finetune_hparams: GLUETrainerHparams) -> None:
    """Logic for handling a pretraining job spawn based on storage and training settings."""
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    training_script = os.path.join(root_dir, 'run_composer_trainer.py')

    # manually copy pretrain_hparams to temporary file
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = os.path.join(tmp_dir.name, 'pretrained_hparams.yaml')
    with open(file) as infile, open(tmp_file, 'w+') as outfile:
        hparams = yaml.load(infile, yaml.Loader)
        pretrain_hparams = hparams['pretrain_hparams']
        yaml.dump(pretrain_hparams, outfile)

    hp = TrainerHparams.create(cli_args=False, f=tmp_file)
    assert hp.train_dataset is not None
    assert hp.train_batch_size is not None
    dataloader = hp.train_dataset.initialize_object(dataloader_hparams=hp.dataloader, batch_size=hp.train_batch_size)
    assert isinstance(dataloader, DataSpec)
    dataloader_len = len(dataloader.dataloader)  # type: ignore
    run_name = hp.run_name
    assert run_name is not None
    assert hp.save_folder is not None
    save_folder = os.path.join(run_name, hp.save_folder)

    if training_scheme == 'all':  # extract run_name from trainer args for finetuning
        # list and save checkpoint paths
        finetune_hparams.save_folder = save_folder
        finetune_hparams.finetune_ckpts = get_ckpt_names(hp, run_name, dataloader_len)

    # call via composer to ensure pretraining done distributedly across all available GPUs
    subprocess.run(args=['composer', training_script, '-f', tmp_file, '--save_folder', save_folder], check=True)


def run_finetuner(training_scheme: str, file: str, save_locally: bool, load_locally: bool, save_folder: str,
                  finetune_hparams) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Logic for handling a finetuning job spawn based on storage and training settings."""
    # set automatic load and save paths
    if load_locally:
        all_ckpts_list = os.listdir(save_folder)
    else:
        all_ckpts_list = finetune_hparams.finetune_ckpts

    # First, fine-tune COLA, MNLI, QNLI, QQP, and SST-2 from every pre-trained checkpoint, saving MNLI checkpoints for the next round
    task_to_save_ckpt = {'cola': False, 'sst-2': False, 'qqp': False, 'qnli': False, 'mnli': True}
    results_a = spawn_finetuning_jobs(task_to_save_ckpt,
                                      all_ckpts_list,
                                      save_folder,
                                      file,
                                      save_locally,
                                      load_locally,
                                      load_ignore_keys=['state/model/model.classifier*'])

    # Second, fine-tune RTE, MRPC, and STS-B from every pre-trained checkpoint's downstream MNLI checkpoints
    # Note: If MNLI ran for multiple seeds, this checkpoint will come from the last MNLI seed to finish.
    ckpt_filenames = [f'{save_folder}/mnli-{idx:03d}/latest-rank0.pt' for idx in range(len(all_ckpts_list))]
    mnli_task_to_save_ckpt = {'rte': False, 'mrpc': False, 'stsb': False}
    results_b = spawn_finetuning_jobs(
        mnli_task_to_save_ckpt,
        ckpt_filenames,
        save_folder,
        file,
        save_locally,
        load_locally=save_locally,
        parent_ckpts=all_ckpts_list,
        load_ignore_keys=['state/model/model.classifier*'],
    )

    return results_a + results_b


def _main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], '--help']

    file = get_args()
    finetune_hparams, training_scheme, save_locally, load_locally = get_finetune_hparams()

    # Pretrain
    if training_scheme in ('pretrain', 'all'):
        run_pretrainer(training_scheme, file, finetune_hparams)
        print('PRETRAINING COMPLETE')

    # Finetune
    if training_scheme in ('finetune', 'all'):
        assert finetune_hparams.save_folder is not None
        results = run_finetuner(training_scheme, file, save_locally, load_locally, finetune_hparams.save_folder,
                                finetune_hparams)
        print('FINETUNING COMPLETE')

        # Process and print the collected results into final GLUE metrics
        glue_task_names = ['cola', 'sst-2', 'qqp', 'qnli', 'mnli', 'rte', 'mrpc', 'stsb']
        glue_metrics = GlueState(glue_task_names, {})
        for task, ckpt, metric in results:
            log_metrics(metric, task, ckpt, glue_metrics)
        print_metrics(glue_metrics)


if __name__ == '__main__':
    _main()
