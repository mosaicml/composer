# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Specifically designed for the NLP use case, allowing pre-training and fine-tuning on
downstream tasks to be handled within one script.

Example that pretrains a BERT:
    >>> composer examples/run_nlp_trainer.py
    -f composer/yamls/models/glue/glue_example.yaml
    --training_scheme pretrain

Example that pretrains and finetunes a BERT:
    >>> composer examples/run_nlp_trainer.py
    -f composer/yamls/models/glue/glue_example.yaml
    --training_scheme all

Example that finetunes a pretrained BERT:

    >>> composer examples/run_nlp_trainer.py
    -f composer/yamls/models/glue/glue_example.yaml
    --training_scheme finetune
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import fields
from typing import List, Optional, Union

import boto3
import torch
import yahp as hp
from tabulate import tabulate

import composer
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.metrics.glue import GlueState
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.nlp_trainer_hparams import GLUETrainerHparams, NLPTrainerHparams
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils.warning_helpers import warning_on_one_line

def init_cuda_queue(queue_size: int) -> mp.Queue:
    """Generate a multiprocessing queue to store queue_size GPU IDs. The multiprocessing package has no way of extracting the worker name from the worker, therefore a queue is used to map pool workers to GPUs to spawn finetune jobs one."""
    cuda_envs = mp.Queue(queue_size)
    cuda_envs_list = range(queue_size)
    for e in cuda_envs_list:
        cuda_envs.put(e)

    return cuda_envs


def init_cuda_env(cuda_envs: mp.Queue) -> None:
    """Set up a single GPU CUDA environment on initialization of a mp process pool."""
    env = cuda_envs.get()
    torch.cuda.set_device(env)

    # fake a single node world
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_WORLD_SIZE'] = '1'


def get_all_s3_checkpoints(bucket_name: str, folder_name: str) -> List[str]:
    """List all the checkpoints in a given S3 bucket and folder."""
    # pulls AWS credentials from environment variables, assumes that aws_session_token is defined
    session = boto3.Session()

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    ckpt_names = []
    for obj in bucket.objects.all():
        if folder_name in obj.key:
            obj_name = obj.key
            ckpt_names.append(obj_name)

    return ckpt_names


def output_metrics(glue_metrics: GlueState) -> None:
    """Consolidate and prettify metrics."""
    tasks = glue_metrics.task_names
    large_tasks = ['mnli', 'qnli', 'qqp', 'sst2']
    # init table headers
    headers = ['Checkpoint']
    headers.extend([f'{task.upper()}' for task in sorted(tasks)])
    headers.extend(['GLUE-Large', 'GLUE-All'])
    tb = [headers]

    # fill table
    for ckpt in glue_metrics.get_logged_ckpts():
        output_line = [ckpt]
        glue_all = 0
        glue_large = 0
        # Per task score
        for task in sorted(glue_metrics.task_names):
            task_metric = 0 if not glue_metrics.get_task_metric(ckpt, task) else glue_metrics.get_task_metric(
                ckpt, task)
            output_line.append('{:.4f}'.format(task_metric))
            glue_all += task_metric
            if task in large_tasks:
                glue_large += task_metric
        # GLUE Large and GLUE All
        output_line.append('{:.4f}'.format(glue_large / len(large_tasks)))
        output_line.append('{:.4f}'.format(glue_all / len(tasks)))
        tb.append(output_line)

    print(tabulate(tb, headers='firstrow'))


def log_metrics(ckpt_filename: str, glue_metrics: GlueState, f_metric: Future) -> None:
    """Callback function for metric collection."""
    metric = f_metric.result()

    if ckpt_filename not in glue_metrics.get_logged_ckpts():
        glue_metrics.init_ckpt_dict(ckpt_filename)

    task = list(metric.keys())[0]
    formatted_task = task.split('_')[1]  # remove "glue_" prefix
    for key in metric.keys():  # handle case where mnli has glue_mnli and glue_mnli_mismatched
        for m in metric[key]:
            if not glue_metrics.get_task_metric(ckpt_filename, formatted_task):
                glue_metrics.set_task_metric(ckpt_filename, formatted_task, metric[key][m].item())
            else:
                old_val = glue_metrics.get_task_metric(ckpt_filename, formatted_task)
                glue_metrics.set_task_metric(ckpt_filename, formatted_task,
                                             sum([old_val, metric[key][m].item()]) / 2)  # automatically average scores


def merge_hparams(hparams: TrainerHparams, override_hparams: Union[GLUETrainerHparams, None]) -> TrainerHparams:
    """Overrides the atttributes of the hparams instance with those of the provided override_hparams."""
    if override_hparams:
        for field in fields(override_hparams):
            if getattr(override_hparams, field.name):
                setattr(hparams, field.name, getattr(override_hparams, field.name))

    return hparams


def setup_finetuning_jobs(finetune_tasks: List[str],
                          save_ckpts: List[bool],
                          ckpt_load_path: str,
                          ckpt_filename: str,
                          ckpt_save_folder: str,
                          glue_metrics: GlueState,
                          base_yaml_file: str,
                          save_locally: bool,
                          parent_ckpt: Optional[str] = None,
                          load_ignore_keys: Optional[List[str]] = None) -> None:
    """Set up CUDA environment and process pool for given finetuning jobs."""
    cuda_envs = init_cuda_queue(torch.cuda.device_count())
    num_tasks = len(finetune_tasks)

    if parent_ckpt:
        wandb_group_name = parent_ckpt
        logged_ckpt_name = parent_ckpt
    else:
        wandb_group_name = ckpt_load_path
        logged_ckpt_name = ckpt_load_path

    # finetuning from pretrained checkpoint
    print(f'FINETUNING ON {ckpt_filename}!')
    log_metrics_lambda = lambda x: log_metrics(f_metric=x, ckpt_filename=logged_ckpt_name, glue_metrics=glue_metrics)
    executor = Pool(max_workers=torch.cuda.device_count(), initializer=init_cuda_env, initargs=(cuda_envs,))
    for rank in range(num_tasks):
        future = executor.submit(train_finetune, base_yaml_file, finetune_tasks[rank], save_ckpts[rank], ckpt_load_path,
                                 wandb_group_name, ckpt_save_folder, save_locally, load_ignore_keys)
        future.add_done_callback(log_metrics_lambda)

    executor.shutdown(wait=True)  # wait for processes and callbacks to complete

    cuda_envs.close()
    cuda_envs.join_thread()


def train_finetune(base_yaml_file: str,
                   task: str,
                   save_ckpt: bool,
                   load_path: str,
                   wandb_group_name: str,
                   save_folder: str,
                   save_locally: bool,
                   load_ignore_keys: Optional[List[str]] = None):
    """Run single instance of a finetuning job on given task."""
    finetune_hparams = NLPTrainerHparams.create(cli_args=False, f=base_yaml_file).finetune_hparams
    task_hparams = TrainerHparams.create(cli_args=False, f=f'./composer/yamls/models/glue/{task}.yaml')

    ft_hparams = merge_hparams(task_hparams, finetune_hparams)

    ft_hparams.load_path = load_path
    ft_hparams.device = DeviceGPU(torch.cuda.current_device())
    ft_hparams.log_to_console = False
    ft_hparams.progress_bar = False

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
                logger._rank_zero_only = True

    # saving single checkpoint at the end of training the task
    if save_ckpt:
        # add task specific artifact logging informatino
        if save_locally:
            task_ckpt_path = os.path.join(save_folder, task)
            ft_hparams.save_folder = task_ckpt_path
            ft_hparams.save_num_checkpoints_to_keep = 1
            ft_hparams.save_overwrite = True

            if not os.path.exists(task_ckpt_path):
                os.mkdir(task_ckpt_path)
        else:
            ft_hparams.save_folder = task
            save_artifact_name = f'{save_folder}/{task}/ep{{epoch}}-ba{{batch}}-rank{{rank}}'
            save_latest_artifact_name = f'{save_folder}/{task}/latest-rank{{rank}}'
            ft_hparams.save_artifact_name = save_artifact_name
            ft_hparams.save_latest_artifact_name = save_latest_artifact_name
            ft_hparams.save_overwrite = True

    print(f'\n --------\n SPAWNING TASK {task.upper()}\n DEVICE: {torch.cuda.current_device()}\n --------')

    trainer = ft_hparams.initialize_object()
    trainer.fit()
    print(f'\nFINISHED TRAINING TASK {task.upper()}\n')
    return trainer.state.current_metrics


def get_args() -> argparse.Namespace:
    """Get NLPTrainerHparams arguments from CLI."""
    parser = hp.get_argparse(NLPTrainerHparams)
    parser.add_argument('--training_scheme',
                        help='training scheme used (one of "pretrain", "finetune", or "all")',
                        choices=['pretrain', 'finetune', 'all'],
                        required=True)

    args = parser.parse_args()

    return args


def validate_args(args: argparse.Namespace, hp: NLPTrainerHparams) -> None:
    """Validate CLI args as well as finetune-specific parameters."""
    if not args.file:
        raise ValueError(
            'must specify a top level file to store shared configs for the entrypoint. see composer/yamls/models/glue/glue_example.yaml for an example.'
        )

    if args.training_scheme == 'finetune' and (not hp.finetune_hparams) or (hp.finetune_hparams and
                                                                            not hp.finetune_hparams.load_path):
        raise ValueError('load_path to checkpoint folder must be specified if finetuning a model')

    elif args.training_scheme == 'all' and hp.finetune_hparams is None:
        warnings.warn('No shared finetune_hparams specified. All finetune tasks will use their default configurations.')
        hp.finetune_hparams = GLUETrainerHparams()

    elif args.training_scheme == 'pretrain' and hp.finetune_hparams:
        warnings.warn('finetune_hparams specified. These values will be ignored during pretraining.')

    elif args.training_scheme == 'all' and hp.finetune_hparams and hp.finetune_hparams.load_path:
        warnings.warn('load_path specified in finetune_hparams. This value will be overriden during finetuning.')


def get_finetune_hparams(args: argparse.Namespace) -> Union[GLUETrainerHparams, None]:
    """Extract finetune-specific hparams from the provided file and add entrypoint specific args to it."""
    hp = NLPTrainerHparams.create(cli_args=False, f=args.file)
    validate_args(args, hp)

    hparams = None
    if args.training_scheme != 'pretrain':
        hparams = hp.finetune_hparams
        args.save_locally = True
        if hparams.loggers:
            for l in hparams.loggers:
                if isinstance(l, ObjectStoreLogger):
                    args.save_locally = False
                    args.bucket = l.object_store.bucket

        if args.training_scheme == 'all':
            hparams.save_folder = hp.pretrain_hparams.save_folder

    return hparams


def run_pretrainer(args) -> None:
    """Logic for handling a pretraining job spawn based on storage and training settings."""
    root_dir = os.path.join(os.path.dirname(composer.__file__), '..')
    training_script = os.path.join(root_dir, 'examples/run_composer_trainer.py')

    # manually copy pretrain_hparams to temporary file (workaround until CO-766 resolved)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = os.path.join(tmp_dir.name, 'pretrained_hparams.yaml')
    with open(args.file) as infile, open(tmp_file, 'w+') as outfile:
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

    if args.training_scheme == 'all':  # extract run_name from trainer args for finetuning
        hp = TrainerHparams.create(cli_args=False, f=tmp_file)
        args.run_name = hp.run_name
        if args.save_locally:
            args.save_folder = hp.save_folder
        else:
            args.save_folder = os.path.join(args.run_name, hp.save_folder)

    proc = subprocess.Popen(['composer', training_script, '-f', tmp_file, '--save_folder', args.save_folder])
    proc.communicate()  # block until training is complete, doesn't interfere with open file streaming


def run_finetuner(args: argparse.Namespace, finetune_hparams, glue_metrics: GlueState) -> None:
    """Logic for handling a finetuning job spawn based on storage and training settings."""
    # set automatic load and save paths
    if args.training_scheme == 'all':  # load checkpoints from where pretraining saved them
        ckpt_folder = args.save_folder
    else:  # finetune only
        ckpt_folder = finetune_hparams.load_path

    if args.save_locally:
        all_ckpts_list = [f for f in os.listdir(ckpt_folder) if not os.path.isdir(os.path.join(ckpt_folder, f))]
    else:
        all_ckpts_list = get_all_s3_checkpoints(args.bucket, ckpt_folder)

    # finetune on every pretrained checkpoint
    for ckpt_filename in all_ckpts_list:
        parent_ckpt = ckpt_filename  # necessary for logging
        if args.save_locally:
            ckpt_load_path = os.path.join(ckpt_folder, ckpt_filename)
            save_locally = True

            # skip symlinks to existing checkpoints
            if (os.path.islink(ckpt_load_path)):
                continue
        else:
            ckpt_load_path = ckpt_filename
            save_locally = False

            # skip symlinks to existing checkpoints
            if ('symlink' in ckpt_filename):
                continue

        finetune_tasks = ['cola', 'sst-2', 'qqp', 'qnli', 'mnli']
        save_ckpts = [False, False, False, False, True]
        setup_finetuning_jobs(finetune_tasks,
                              save_ckpts,
                              ckpt_load_path,
                              ckpt_filename,
                              ckpt_folder,
                              glue_metrics,
                              args.file,
                              save_locally=save_locally,
                              parent_ckpt=parent_ckpt,
                              load_ignore_keys=['state/model/model.classifier*'])

        # finetune on inference tasks using last mnli checkpoint
        if args.save_locally:
            # load checkpoints from where mnli saved them
            ft_ckpt_folder = os.path.join(ckpt_folder, 'mnli')
            ckpt_filename = sorted(os.listdir(ft_ckpt_folder))[0]
            ckpt_load_path = os.path.join(ft_ckpt_folder, ckpt_filename)
            save_locally = True
        else:
            ckpt_filename = get_all_s3_checkpoints(args.bucket, f'{ckpt_folder}/mnli')[-1]  # get last checkpoint
            ckpt_load_path = ckpt_filename
            save_locally = False

        mnli_finetune_tasks = ['rte', 'mrpc', 'stsb']
        save_ckpts = [False, False, False]
        # delete finetuning head to reinitialize number of classes
        setup_finetuning_jobs(
            mnli_finetune_tasks,
            save_ckpts,
            ckpt_load_path,
            ckpt_filename,
            finetune_hparams.save_folder,
            glue_metrics,
            args.file,
            save_locally=save_locally,
            parent_ckpt=parent_ckpt,
            load_ignore_keys=['state/model/model.classifier*'],
        )  # type: ignore


def _main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], '--help']

    args = get_args()
    finetune_hparams = get_finetune_hparams(args)

    # Pretrain
    if args.training_scheme != 'finetune':
        run_pretrainer(args)
        print('PRETRAINING COMPLETE')  #FIXME: LOG

    # Finetune
    glue_task_names = ['cola', 'sst2', 'qqp', 'qnli', 'mnli', 'rte', 'mrpc', 'stsb']
    glue_metrics = GlueState(glue_task_names, {})
    if args.training_scheme != 'pretrain':
        run_finetuner(args, finetune_hparams, glue_metrics)
        print('FINETUNING COMPLETE')

    # output GLUE metrics if finetuning
    if args.training_scheme != 'pretrain':
        output_metrics(glue_metrics)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.environ['WANDB_START_METHOD'] = 'thread'  # to not interfere with multiprocessing
    _main()
