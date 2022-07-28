# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Specifically designed for the NLP use case, allowing pre-training and fine-tuning on
downstream tasks to be handled within one script.

Example that pretrains a BERT::
    >>> composer examples/run_nlp_trainer.py
    -f composer/yamls/models/glue/glue_example.yaml
    --training_scheme pretrain

Example that pretrains and finetunes a BERT::
    >>> composer examples/run_nlp_trainer.py
    -f composer/yamls/models/glue/glue_example.yaml
    --training_scheme all

Example that finetunes a pretrained BERT::

    >>> composer examples/run_nlp_trainer.py
    -f composer/yamls/models/glue/glue_example.yaml
    --training_scheme finetune
"""
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from composer.cli.launcher import _get_free_tcp_port
from composer.core.time import Time, TimeUnit, Timestamp
from composer.utils.file_helpers import format_name_with_dist_and_time
import yahp as hp
from tabulate import tabulate

import composer
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.nlp_trainer_hparams import GLUETrainerHparams, NLPTrainerHparams
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils.misc import warning_on_one_line


class GLUEMetricsState:
    """Class mapping all GLUE tasks to their respective average metric values."""

    def __init__(self, task_names: List[str]) -> None:
        self.task_to_avg_metric = {}

        for task in task_names:
            self.task_to_avg_metric[task] = None


@dataclass
class GlueState:
    """Class storing all GLUE metrics per checkpoint collected during a finetuning job spawned by the NLP entrypoint.

    This class maps checkpoint names to GLUEMetricsState instances which map tasks to their respective average
    metric values.

    Args:
        task_names list(str): the names of the GLUE tasks stored in the data struct
        ckpt_to_tasks dict(str, GLUEMetricsState): dictionary mapping checkpoint names to GLUEMetricsState instances
    """
    task_names: List[str]
    ckpt_to_tasks: Dict[str, GLUEMetricsState]


def init_cuda_queue(queue_size: int) -> mp.Queue:
    """Generate a multiprocessing queue to store queue_size GPU IDs. The multiprocessing package has no way of extracting the worker ID from the worker name; therefore, a queue is used to map pool workers to GPUs to spawn finetune jobs one."""
    cuda_envs = mp.Queue(queue_size)
    cuda_envs_list = range(queue_size)
    for e in cuda_envs_list:
        cuda_envs.put(e)

    return cuda_envs


def init_cuda_env(cuda_envs: mp.Queue, free_port: int) -> None:
    """Set up a single GPU CUDA environment on initialization of a mp process pool."""
    env = cuda_envs.get()
    torch.cuda.set_device(env)

    # fake a single node world
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_WORLD_SIZE'] = '1'
    os.environ['NODE_RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = '0'


def print_metrics(glue_metrics: GlueState) -> None:
    """Consolidate and prettify metrics."""
    tasks = glue_metrics.task_names
    large_tasks = ['mnli', 'qnli', 'qqp', 'sst2']
    # init table headers
    headers = ['Checkpoint']
    headers.extend([f'{task.upper()}' for task in sorted(tasks)])
    headers.extend(['GLUE-Large', 'GLUE-All'])
    tb = [headers]

    # fill table
    for ckpt in glue_metrics.ckpt_to_tasks.keys():
        output_line = [ckpt]
        glue_all = 0
        glue_large = 0
        # Per task score
        for task in sorted(glue_metrics.task_names):
            task_metric = glue_metrics.ckpt_to_tasks[ckpt].task_to_avg_metric[task]
            logged_metric = 0
            if task_metric:
                logged_metric = sum(task_metric) / len(glue_metrics.task_names)  # average all metrics
            output_line.append('{:.4f}'.format(logged_metric))
            glue_all += logged_metric
            if task in large_tasks:
                glue_large += logged_metric
        # GLUE Large and GLUE All
        output_line.append('{:.4f}'.format(glue_large / len(large_tasks)))
        output_line.append('{:.4f}'.format(glue_all / len(tasks)))
        tb.append(output_line)

    print(tabulate(tb, headers='firstrow'))


def log_metrics(metric: Dict[str, Dict], ckpt_filename: str, glue_metrics: GlueState) -> None:
    """Callback function for metric collection.

    Args:
        metric (Dict): Metrics returned from ``train_finetune()`` for a given GLUE task
        ckpt_filename (str): Checkpoint to log metrics under
        glue_metrics (GlueState): GlueState object storing all the glue metrics for the entrypoint's current run
    """
    if ckpt_filename not in glue_metrics.ckpt_to_tasks.keys():
        glue_metrics.ckpt_to_tasks[ckpt_filename] = GLUEMetricsState(glue_metrics.task_names)

    task = list(metric.keys())[0]
    formatted_task = task.split('_')[1]  # remove "glue_" prefix
    for metrics in metric.values():  # handle case where mnli has glue_mnli and glue_mnli_mismatched
        for metric_val in metrics.values():  # handle case where mnli has glue_mnli and glue_mnli_mismatched
            tasks = glue_metrics.ckpt_to_tasks[ckpt_filename]
            task_metric = tasks.task_to_avg_metric[formatted_task]
            if not task_metric:
                glue_metrics.ckpt_to_tasks[ckpt_filename].task_to_avg_metric[formatted_task] = []
            glue_metrics.ckpt_to_tasks[ckpt_filename].task_to_avg_metric[formatted_task].append(metric_val.item())


def merge_hparams(hparams: TrainerHparams, override_hparams: Optional[GLUETrainerHparams]) -> TrainerHparams:
    """Overrides the atttributes of the hparams instance with those of the provided override_hparams."""
    
    if override_hparams:
        hparams.algorithms = override_hparams.algorithms if override_hparams.algorithms else hparams.algorithms
        hparams.load_ignore_keys = override_hparams.load_ignore_keys if override_hparams.load_ignore_keys else hparams.load_ignore_keys
        hparams.load_path = override_hparams.load_path if override_hparams.load_path else hparams.load_path
        hparams.load_object_store = override_hparams.load_object_store if override_hparams.load_object_store else hparams.load_object_store
        hparams.loggers = override_hparams.loggers if override_hparams.loggers else hparams.loggers
        hparams.save_folder = override_hparams.save_folder if override_hparams.save_folder else hparams.save_folder
      
    return hparams


def spawn_finetuning_jobs(
    task_to_save_ckpt: Dict[str, bool],
    ckpt_load_path: str,
    ckpt_save_folder: str,
    glue_metrics: GlueState,
    base_yaml_file: str,
    save_locally: bool,
    parent_ckpt: Optional[str] = None,
    load_ignore_keys: Optional[List[str]] = None,
) -> None:
    """Set up CUDA environment and process pool for given finetuning jobs."""
    cuda_envs = init_cuda_queue(torch.cuda.device_count())
    finetune_tasks = list(task_to_save_ckpt.keys())
    num_tasks = len(finetune_tasks)

    if parent_ckpt:
        wandb_group_name = parent_ckpt
        logged_ckpt_name = parent_ckpt
    else:
        wandb_group_name = ckpt_load_path
        logged_ckpt_name = ckpt_load_path

    # finetuning from pretrained checkpoint
    print(f'FINETUNING ON {ckpt_load_path}!')
    done_callback = lambda future: log_metrics(
        metric=future.result(), ckpt_filename=logged_ckpt_name, glue_metrics=glue_metrics)
    free_port = _get_free_tcp_port()
    executor = Pool(max_workers=torch.cuda.device_count(), initializer=init_cuda_env, initargs=(cuda_envs, free_port))
    for rank in range(num_tasks):
        task = finetune_tasks[rank]
        future = executor.submit(train_finetune, base_yaml_file, task, task_to_save_ckpt[task], ckpt_load_path,
                                 wandb_group_name, ckpt_save_folder, save_locally, free_port + rank, load_ignore_keys)
        future.add_done_callback(done_callback)

    executor.shutdown(wait=True)  # wait for processes and callbacks to complete

    cuda_envs.close()
    cuda_envs.join_thread()

def to_cpu(gpu_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Copy given dictionary to CPU to avoid multiprocessing GPU tensor pickling issues."""
    cpu_dict = {} 
    for key in gpu_dict.keys():
        if key not in cpu_dict.keys():
            cpu_dict[key] = {}
        for inner_key in gpu_dict[key].keys():
            cpu_dict[key][inner_key] = gpu_dict[key][inner_key].cpu() 

    return cpu_dict

def train_finetune(
    base_yaml_file: str,
    task: str,
    save_ckpt: bool,
    load_path: str,
    wandb_group_name: str,
    save_folder: str,
    save_locally: bool,
    master_port: int,
    load_ignore_keys: Optional[List[str]] = None,
):
    """Run single instance of a finetuning job on given task."""
    os.environ['MASTER_PORT'] = f'{master_port}' # set unique master port for each spawn

    finetune_hparams = NLPTrainerHparams.create(cli_args=False, f=base_yaml_file).finetune_hparams
    task_hparams = TrainerHparams.create(cli_args=False, f=f'./composer/yamls/models/glue/{task}.yaml')

    ft_hparams = merge_hparams(task_hparams, finetune_hparams)
    
    ft_hparams.load_path = load_path
    ft_hparams.device = DeviceGPU(torch.cuda.current_device())
    ft_hparams.log_to_console = False
    ft_hparams.progress_bar = False
    ft_hparams.save_overwrite = True

    if ft_hparams.load_ignore_keys:
        ft_hparams.load_ignore_keys.extend(load_ignore_keys)
    else:
        ft_hparams.load_ignore_keys = load_ignore_keys

    # add finetune-specific tags to wandb if logger exists
    if ft_hparams.loggers:
        for logger in ft_hparams.loggers:
            if isinstance(logger, WandBLogger):
                if not logger._init_kwargs['tags']:
                    logger._init_kwargs['tags'] = []
                logger._init_kwargs['tags'].append(task)
                logger._init_kwargs['group'] = wandb_group_name

    # saving single checkpoint at the end of training the task
    if save_ckpt:
        # add task specific artifact logging information
        ft_hparams.save_folder = f'{save_folder}/{task}' 
        save_artifact_name = f'{save_folder}/{task}/ep{{epoch}}-ba{{batch}}-rank{{rank}}' # ignored if not uploading
        save_latest_artifact_name = f'{save_folder}/{task}/latest-rank{{rank}}'
        ft_hparams.save_artifact_name = save_artifact_name
        ft_hparams.save_latest_artifact_name = save_latest_artifact_name

        ft_hparams.save_num_checkpoints_to_keep = 0

        if save_locally:
            if not os.path.exists(ft_hparams.save_folder):
                os.mkdir(ft_hparams.save_folder)
            ft_hparams.save_num_checkpoints_to_keep = 1
           
    print(f'\n --------\n SPAWNING TASK {task.upper()}\n DEVICE: {torch.cuda.current_device()}\n --------')

    trainer = ft_hparams.initialize_object()
    trainer.fit()
    print(f'\nFINISHED TRAINING TASK {task.upper()}\n')
    return to_cpu(trainer.state.current_metrics)

def get_args() -> Tuple:
    """Get NLPTrainerHparams arguments from CLI."""
    parser = hp.get_argparse(NLPTrainerHparams)
    args = parser.parse_args()

    return args.file, args.training_scheme


def validate_args(training_scheme: str, hp: NLPTrainerHparams) -> None:
    """Validate CLI args as well as finetune-specific parameters."""
    if training_scheme == 'finetune' and ((not hp.finetune_hparams) or
                                               (hp.finetune_hparams and not hp.finetune_hparams.finetune_ckpts)):
        raise ValueError('load_path to checkpoints must be specified if finetuning a model')

    elif training_scheme == 'all' and hp.finetune_hparams is None:
        warnings.warn('No shared finetune_hparams specified. All finetune tasks will use their default configurations.')

    elif training_scheme == 'pretrain' and hp.finetune_hparams:
        warnings.warn('finetune_hparams specified. These values will be ignored during pretraining.')

    elif training_scheme == 'all' and hp.finetune_hparams and hp.finetune_hparams.finetune_ckpts:
        warnings.warn('finetune_ckpts specified in finetune_hparams. This value will be overriden during finetuning.')


def get_finetune_hparams(training_scheme: str) -> Tuple:
    """Extract finetune-specific hparams from the provided file and add entrypoint specific args to it."""
    hp = NLPTrainerHparams.create()
    validate_args(training_scheme, hp)

    hparams = None
    save_locally = True
    if training_scheme != 'pretrain':
        hparams = hp.finetune_hparams
        if not hparams:
            hparams = GLUETrainerHparams()
        if hparams.loggers:
            for l in hparams.loggers:
                if isinstance(l, ObjectStoreLogger):
                    save_locally = False

    return hparams, save_locally

def get_ckpt_names(hp: TrainerHparams, run_name: str, dataloader_len: int) -> List[str]:
    """Extract list of checkpoints that will be saved by the given configuration."""
    ckpt_names = []
    interval = Time.from_timestring(hp.save_interval)
    duration = Time.from_timestring(hp.max_duration)

    ep = 0
    ba = 0
    loop = True
    save = False
    while loop:
        if save:
            time = Timestamp(epoch=ep, batch=ba)
            formatted_ckpt_name = format_name_with_dist_and_time(hp.save_artifact_name, run_name, time)
            ckpt_names.append(formatted_ckpt_name)
            save = False

        if interval.unit == TimeUnit.BATCH:
            save = True
        if ba >= dataloader_len: # batches per epoch
            ep += 1
            if interval.unit == TimeUnit.EPOCH: 
                save = True

        ba += interval.value
        if duration.unit == TimeUnit.BATCH:
            if ba >= duration.value: loop = False
        elif duration.unit == TimeUnit.EPOCH:
            if ep >= duration.value: loop = False
        elif duration.unit == TimeUnit.SAMPLE:
            if ba * hp.train_batch_size >= duration.value: 
                loop = False
                if ba * hp.train_batch_size > duration.value: # don't save last batch
                    save = False 
        
    return ckpt_names

def run_pretrainer(training_scheme: str, file: str, save_locally: bool, finetune_hparams: GLUETrainerHparams) -> None:
    """Logic for handling a pretraining job spawn based on storage and training settings."""
    root_dir = os.path.join(os.path.dirname(composer.__file__), '..')
    training_script = os.path.join(root_dir, 'examples/run_composer_trainer.py')

    # manually copy pretrain_hparams to temporary file (workaround until CO-766 resolved) #TODO: MERGE DEV INTO THIS AND FIX AFTER PUSH 
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = os.path.join(tmp_dir.name, 'pretrained_hparams.yaml')
    with open(file) as infile, open(tmp_file, 'w+') as outfile:
        copy = False
        for line in infile:
            if line.strip() == 'pretrain_hparams:':
                copy = True
                continue
            elif line.strip() == 'finetune_hparams:':
                copy = False
                continue
            elif copy:
                outfile.write(line)
    
    hp = TrainerHparams.create(cli_args=False, f=tmp_file)
    trainer = hp.initialize_object()
    run_name = hp.run_name
    save_folder = os.path.join(run_name, hp.save_folder) 

    if training_scheme == 'all':  # extract run_name from trainer args for finetuning
        # list and save checkpoint paths 
        finetune_hparams.save_folder = save_folder
        finetune_hparams.finetune_ckpts = get_ckpt_names(hp, run_name, trainer.state.dataloader_len.value)
        
    subprocess.run(args=['composer', training_script, '-f', tmp_file, '--save_folder', save_folder], check=True)

def run_finetuner(training_scheme: str, file: str, save_locally: bool, save_folder: str, finetune_hparams, glue_metrics: GlueState) -> None:
    """Logic for handling a finetuning job spawn based on storage and training settings."""
    # set automatic load and save paths
    all_ckpts_list = finetune_hparams.finetune_ckpts

    # finetune on every pretrained checkpoint
    for ckpt_filename in all_ckpts_list:
        parent_ckpt = ckpt_filename  # necessary for logging

        task_to_save_ckpt = {'cola': False, 'sst-2': False, 'qqp': False, 'qnli': False, 'mnli': True} 
        spawn_finetuning_jobs(task_to_save_ckpt,
                              ckpt_filename,
                              save_folder,
                              glue_metrics,
                              file,
                              save_locally=save_locally,
                              parent_ckpt=parent_ckpt,
                              load_ignore_keys=['state/model/model.classifier*'])

        # finetune on inference tasks using last mnli checkpoint
        ckpt_filename = f'{save_folder}/mnli/latest-rank0'   

        mnli_task_to_save_ckpt = {'rte': False, 'mrpc': False, 'stsb': False}
        # delete finetuning head to reinitialize number of classes
        spawn_finetuning_jobs(
            mnli_task_to_save_ckpt,
            ckpt_filename,
            save_folder,
            glue_metrics,
            file,
            save_locally=save_locally,
            parent_ckpt=parent_ckpt,
            load_ignore_keys=['state/model/model.classifier*'],
        )


def _main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], '--help']

    file, training_scheme = get_args()
    finetune_hparams, save_locally = get_finetune_hparams(training_scheme)

    # Pretrain
    if training_scheme != 'finetune':
        run_pretrainer(training_scheme, file, save_locally, finetune_hparams)
        print('PRETRAINING COMPLETE')

    # Finetune
    glue_task_names = ['cola', 'sst2', 'qqp', 'qnli', 'mnli', 'rte', 'mrpc', 'stsb']
    glue_metrics = GlueState(glue_task_names, {})
    if training_scheme != 'pretrain':
        run_finetuner(training_scheme, file, save_locally, finetune_hparams.save_folder, finetune_hparams, glue_metrics)
        print('FINETUNING COMPLETE')

        # output GLUE metrics
        print_metrics(glue_metrics)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    _main()
