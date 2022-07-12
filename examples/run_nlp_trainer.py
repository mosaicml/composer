# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.
Specifically designed for the NLP use case, allowing pre-training and fine-tuning on 
downstream tasks to be handled within one script.

Adds a --datadir flag to conveniently set a common
data directory for both train and validation datasets.

Example that pretrains a BERT:
    >>> composer examples/run_nlp_trainer.py 
    -f composer/yamls/models/bert-base.yaml 
    --training_scheme pretrain 

Example that pretrains and finetunes a BERT:
    >>> composer examples/run_nlp_trainer.py 
    -f composer/yamls/models/bert-base.yaml 
    --training_scheme all 
    --save_folder ~/checkpoints

Example that finetunes a pretrained BERT:

    >>> composer examples/run_nlp_trainer.py
    -f composer/yamls/models/bert_base.yaml
    --training_scheme finetune
    --load_path ~/checkpoints
"""
import argparse
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import warnings
from typing import List, Optional, Type

import boto3
import torch
import yahp as hp
from tabulate import tabulate

import composer
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils.object_store.object_store_hparams import S3ObjectStoreHparams
from composer.utils.object_store.s3_object_store import S3ObjectStore

# TODO: wandb integrations, reset all configs and training lengths back to default
# track GLUE Metrics
GLUE_METRICS = {}

def _warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'

''' Generate a multiprocessing queue to store queue_size GPU IDs. '''
def init_cuda_queue(queue_size: int) -> mp.Queue:
    cuda_envs = mp.Queue(queue_size)
    cuda_envs_list = range(queue_size)
    for e in cuda_envs_list:
        cuda_envs.put(e)

    return cuda_envs

''' Set up a single GPU CUDA environment on initialization of a mp process pool. '''
def init_cuda_env(cuda_envs) -> None:
    try:
        env=cuda_envs.get()
        torch.cuda.set_device(env)
    except (BrokenPipeError, IOError):
        pass

    # fake a single node world 
    os.environ['WORLD_SIZE'] = "1"
    os.environ['LOCAL_WORLD_SIZE'] = "1"

''' List all the checkpoints in a given S3 bucket and folder. ''' 
def get_all_s3_checkpoints(bucket_name: str, folder_name: str) -> List[str]:
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

''' Consolidate and prettify metrics. '''
def output_metrics() -> None:
    tasks = GLUE_METRICS[list(GLUE_METRICS.keys())[0]].keys()
    large_tasks = ['MNLI', 'QNLI', 'QQP', 'SST-2']
    # init table headers 
    headers = ['Checkpoint']
    headers.extend([f"{task.upper()}" for task in tasks])
    headers.extend(['GLUE-Large', 'GLUE-All'])
    tb = [headers]

    # fill table 
    for ckpt in GLUE_METRICS.keys():
        output_line = [ckpt]
        glue_all = 0
        glue_large = 0
        # Per task score
        for task in sorted(GLUE_METRICS[ckpt].keys()):
            output_line.append("{:.2f}".format(GLUE_METRICS[ckpt][task]))
            glue_all += GLUE_METRICS[ckpt][task]
            if task in large_tasks:
                glue_large += GLUE_METRICS[ckpt][task]
        # GLUE Large and GLUE All
        output_line.append("{:.2f}".format(glue_large/len(large_tasks)))
        output_line.append("{:.2f}".format(glue_all/len(tasks)))
        tb.append(output_line)
    
    print(tabulate(tb, headers='firstrow'))

''' Set up CUDA environment and process pool for given finetuning jobs. '''
def setup_finetuning_jobs(finetune_tasks: List[str], save_ckpts: List[bool], ckpt_load_path: str, ckpt_filename: str,
                                         save_folder: str, save_locally: bool, bucket: Optional[str] = None, parent_ckpt: Optional[str] = None, load_ignore_keys: Optional[List[str]] = None) -> None:
    cuda_envs = init_cuda_queue(len(finetune_tasks))
    num_tasks = len(finetune_tasks)
    
    ''' Callback function for metric collection. '''
    def log_metrics(metric) -> None:
        if parent_ckpt: 
            logged_ckpt_name = parent_ckpt
        else:
            logged_ckpt_name = ckpt_filename

        if logged_ckpt_name not in GLUE_METRICS.keys():
                GLUE_METRICS[logged_ckpt_name] = {}
        task = list(metric.keys())[0]
        for m in metric[task]:
            formatted_task = list(metric.keys())[0].split('_')[1] # remove "glue_" prefix
            if formatted_task not in GLUE_METRICS[logged_ckpt_name].keys():
                GLUE_METRICS[logged_ckpt_name][formatted_task] = metric[task][m].item()
            else:
                GLUE_METRICS[logged_ckpt_name][formatted_task] = sum([GLUE_METRICS[logged_ckpt_name][formatted_task], metric[task][m].item()]) / 2  # automatically average scores 
        
    # finetuning from pretrained checkpoint
    print(f"FINETUNING ON {ckpt_filename}!")
    pool = mp.Pool(num_tasks, initializer=init_cuda_env, initargs=(cuda_envs,))
    subprocs = []
    for rank in range(num_tasks):
        sp = pool.apply_async(train_finetune, (finetune_tasks[rank], save_ckpts[rank], ckpt_load_path, save_folder, save_locally, rank, bucket, load_ignore_keys), callback=log_metrics)
        subprocs.append(sp)
    for sp in subprocs:
        sp.wait()
    pool.close()
    pool.join() # wait for join to proceed
   
    cuda_envs.close() 
    cuda_envs.join_thread()
    
'''Run single instance of a finetuning job on given task. ''' 
def train_finetune(task: str, save_ckpt: bool, load_path: str, save_folder: str, save_locally: bool, gpu_id: int, bucket: Optional[str] = None, load_ignore_keys: Optional[List[str]] = None):
    # add dummy arg because arg parser removes first argument in tokenization
    ft_hparams = TrainerHparams.create(cli_args=['-f',  f'./composer/yamls/models/glue/{task}.yaml',
                '--load_path', load_path,
                '--device_id' , f"{gpu_id}",
                "--log_to_console", "False",
                "--progress_bar", "False",
                "--load_ignore_keys", f"{load_ignore_keys}"])
    
    # load checkpoint to finetune on via cloud 
    if not save_locally:
        ft_hparams.load_object_store = S3ObjectStoreHparams(bucket=bucket)

    # saving single checkpoint at the end of training the task 
    if save_ckpt:
        if save_locally:
            task_ckpt_path = os.path.join(save_folder, f'{task}')
            ft_hparams.save_folder = task_ckpt_path
            ft_hparams.save_num_checkpoints_to_keep = 1
            ft_hparams.save_overwrite = True

            if not os.path.exists(task_ckpt_path):
                os.mkdir(task_ckpt_path)
        else:
            ft_hparams.loggers = [ObjectStoreLogger(object_store_cls=S3ObjectStore, object_store_kwargs={'bucket': bucket}, use_procs=False)]
            ft_hparams.save_folder = f'{task}'
            ft_hparams.save_num_checkpoints_to_keep = 0
            save_artifact_name = f'{{run_name}}/{task}/ep{{epoch}}-ba{{batch}}-rank{{rank}}'
            save_latest_artifact_name = f'{{run_name}}/{task}/latest-rank{{rank}}'
            ft_hparams.save_artifact_name = save_artifact_name
            ft_hparams.save_latest_artifact_name = save_latest_artifact_name
            ft_hparams.save_overwrite = True

    print(f" --------\n SPAWNING NEW TASK: {task}\n DEVICE: {torch.cuda.current_device()}\n --------")

    trainer = ft_hparams.initialize_object()
    trainer.fit(duration='3ep')
    print(f"\nFINISHED TRAINING TASK {task.upper()}\n")
    return trainer.state.current_metrics

'''Get the run name by pre-constructing a dummy TrainerHparams instance. ''' 
def get_run_name():
    hp = TrainerHparams.create(cli_args=['--file', 'composer/yamls/models/bert-base.yaml'])
    return hp.run_name

''' Validate NLP specific entrypoint arguments. ''' 
def validate_args(args) -> None:
    if args.training_scheme != 'pretrain' and args.training_scheme != 'finetune' and args.training_scheme != 'all':
            raise ValueError('training_scheme must be one of "pretrain," "finetune," or "all"')

    if args.training_scheme == 'finetune' and args.load_path is None: 
        raise ValueError('load_path to checkpoint folder must be specified if finetuning a model')

    if not args.save_locally and args.bucket is None:
        raise ValueError('S3 bucket name must be specified if saving to cloud')

''' Get TrainerHparams arguments from CLI and add NLP entrypoint specific arguments. ''' 
def get_args():
    parser = hp.get_argparse(TrainerHparams)
    parser.add_argument('--training_scheme', help='training scheme used (one of "pretrain", "finetune", or "all")')
    parser.add_argument('--save_locally', action=argparse.BooleanOptionalAction, help='save to local directory or to cloud using object store if --no-save-locally is enabled')
    parser.add_argument('--bucket', help='S3 bucket name to pull from/save to if --no-save_locally is enabled')
    
    args = parser.parse_args()
    validate_args(args)
    return args

''' Logic for handling a pretraining job spawn based on storage and training settings.'''
def run_pretrainer(args):
    root_dir = os.path.join(os.path.dirname(composer.__file__), '..')
    training_script = os.path.join(root_dir, 'examples/run_composer_trainer.py')
    if args.save_locally:
        args.save_folder = os.path.join(args.run_name, 'checkpoints')
        if args.training_scheme == 'all':
            new_save_folder = os.path.join(args.save_folder, 'pretrained')
        else: # pretrain only
            new_save_folder = args.save_folder
        
        proc = subprocess.Popen(['composer', training_script, '-f', f"{args.file}", '--save_folder', f"{new_save_folder}"], cwd=root_dir)
        
    # saving on cloud
    else:
        if args.training_scheme == 'all':
            args.save_folder = 'pretrained'
            save_artifact_name = '{run_name}/pretrained/ep{epoch}-ba{batch}-rank{rank}-wct{total_wct}'
            proc = subprocess.Popen(['composer', training_script, '-f', f"{args.file}" , '--save_folder', args.save_folder, '--save_artifact_name', save_artifact_name, 
                                '--save_num_checkpoints_to_keep', '0', '--loggers' , 'object_store', '--object_store_hparams', 's3', '--bucket', args.bucket], cwd=root_dir)                
        
        # only pretraining, save to default artifact name 
        else: 
            proc = subprocess.Popen(['composer', training_script, '-f', f"{args.file}", '--save_num_checkpoints_to_keep', '0',
                    '--loggers' , 'object_store', '--object_store_hparams', 's3', '--bucket', args.bucket], cwd=root_dir)
    proc.wait() # block until training is complete 

''' Logic for handling a finetuning job spawn based on storage and training settings.'''
def run_finetuner(args):
    # set automatic load and save paths 
    if args.save_locally:
        if args.training_scheme == 'all':
            # load checkpoints from where pretraining saved them
            ckpt_folder = os.path.join(args.save_folder, 'pretrained')
            save_folder = args.save_folder
        else: # finetune only
            ckpt_folder = args.load_path
            save_folder = os.path.join(args.load_path, '..')

        all_ckpts_list =  sorted(os.listdir(ckpt_folder))
            
    else:
        if args.training_scheme == 'all':
            ckpt_folder = f'{args.run_name}/{args.save_folder}'
            save_folder = args.save_folder
        else: # finetune only
            ckpt_folder = args.load_path # load path should be full/path/to/folder/ containing all ckpts 
            save_folder = args.run_name 

        all_ckpts_list = get_all_s3_checkpoints(args.bucket, ckpt_folder)

    # finetune on every pretrained checkpoint  
    for ckpt_filename in all_ckpts_list:
        if args.save_locally:
            ckpt_load_path = os.path.join(ckpt_folder, ckpt_filename)
            save_locally = True
            # skip symlinks to existing checkpoints
            if (os.path.islink(ckpt_load_path)):
                continue 
        else:
            ckpt_load_path = ckpt_filename
            save_locally = False
            
        # TODO: PUT SEEDS BACK IN
        finetune_tasks = ['cola', 'sst-2', 'qqp', 'qnli', 'mnli']
        save_ckpts = [False, False, False, False, True]
        setup_finetuning_jobs(finetune_tasks, save_ckpts, ckpt_load_path, ckpt_filename, save_folder, save_locally=save_locally, bucket=args.bucket)

        # finetune on inference tasks using last mnli checkpoint 
        parent_ckpt = ckpt_filename # necessary for logging 
        if args.save_locally:
            # load checkpoints from where mnli saved them
            ft_ckpt_folder = os.path.join(save_folder, 'mnli')  
            ckpt_filename =  sorted(os.listdir(ft_ckpt_folder))[0]
            ckpt_load_path = os.path.join(ft_ckpt_folder, ckpt_filename)
            save_locally = True
        else:
            ckpt_filename = get_all_s3_checkpoints(args.bucket, f'{args.run_name}/mnli')[-1] # get latest checkpoint
            ckpt_load_path = ckpt_filename
            save_locally = False

        mnli_finetune_tasks = ['rte', 'mrpc', 'stsb']
        save_ckpts = [False, False, False]
        # delete finetuning head to reinitialize number of classes 
        setup_finetuning_jobs(mnli_finetune_tasks, save_ckpts, ckpt_load_path, ckpt_filename, args.save_folder, args.bucket, parent_ckpt=parent_ckpt, 
                            load_ignore_keys="state/model/model.classifier*", save_locally=save_locally)  # type: ignore

def _main() -> None:
    warnings.formatwarning = _warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], '--help']
    
    args = get_args()
    args.run_name = get_run_name()

    # Pretrain
    if args.training_scheme != 'finetune': 
        run_pretrainer(args)
        print("PRETRAINING COMPLETE") #FIXME: LOG

    # Finetune 
    if args.training_scheme != 'pretrain':
        run_finetuner(args)
        print('FINETUNING COMPLETE')

    # output GLUE metrics if finetuning
    if args.training_scheme != 'pretrain':
        output_metrics()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    _main()

