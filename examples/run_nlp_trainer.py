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
import os
import shutil
import signal
import subprocess
import sys
import warnings
import torch
import multiprocessing as mp
from typing import List, Optional, Type
import composer

# from composer.loggers.logger import LogLevel
from composer.trainer.nlp_hparams import NLPTrainerHparams
from argparse import ArgumentParser
from tabulate import tabulate

# TODO: wandb integrations, refactor everything out of local storage to object store , reset all configs and training lengths back to default
# track GLUE Metrics
GLUE_METRICS = {}

def _warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'

def output_metrics(output_type: str) -> None:
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

def train_finetune_wrapper(finetune_tasks: List[str], save_ckpts: List[bool], ckpt_load_path: str, ckpt_filename: str,
                                         save_folder: str, parent_ckpt: Optional[str] = None, load_ignore_keys: Optional[List[str]] = None) -> None:
    cuda_envs = init_cuda_queue(len(finetune_tasks))
    num_tasks = len(finetune_tasks)
    
    # callback func to store metric collection
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
        sp = pool.apply_async(train_finetune, (finetune_tasks[rank], save_ckpts[rank], ckpt_load_path, save_folder, rank, load_ignore_keys), callback=log_metrics)
        subprocs.append(sp)
    for sp in subprocs:
        sp.wait()
    pool.close()
    pool.join() # wait for join to proceed
   
    cuda_envs.close() 
    cuda_envs.join_thread()
    
def train_finetune(task: str, save_ckpt: bool, load_path: str, save_folder: str, gpu_id: int, load_ignore_keys: Optional[List[str]] = None):
    # add dummy arg because arg parser removes first argument in tokenization
    sys.argv = ['dummy_arg' , '-f', f'./composer/yamls/models/glue/{task}.yaml', 
                '--load_path', load_path,
                '--device_id' , f"{gpu_id}",
                "--log_to_console", "True",
                "--progress_bar", "False",
                "--loggers", "file",
                "--filename" , "{run_name}/logs_{rank}_" + f"{task}" + ".txt",
                "--load_ignore_keys", f"{load_ignore_keys}"]
    # TODO:  use args instead of manually entering these 
    if save_ckpt:
        sys.argv.append("--save_folder")
        task_ckpt_path = os.path.join(save_folder, f'{task}')
        sys.argv.append(task_ckpt_path)
        sys.argv.append("--save_num_checkpoints_to_keep")
        sys.argv.append("1")

        if not os.path.exists(task_ckpt_path):
            os.mkdir(task_ckpt_path)
    
    # set gpu
    print(f" --------\n SPAWNING NEW TASK: {task}\n DEVICE: {torch.cuda.current_device()}\n --------")

    # last file in folder is the symlink to the latest ckpt file 
    ft_hparams = NLPTrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    trainer = ft_hparams.initialize_object()
    trainer.fit(duration='3ep')
    print(f"\nFINISHED TRAINING TASK {task.upper()}\n")
    return trainer.state.current_metrics

def init_cuda_queue(queue_size: int) -> mp.Queue:
    cuda_envs = mp.Queue(queue_size)
    cuda_envs_list = range(queue_size)
    for e in cuda_envs_list:
        cuda_envs.put(e)

    return cuda_envs

def init_cuda_env(cuda_envs) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        env=cuda_envs.get()
        torch.cuda.set_device(env)
    except (BrokenPipeError, IOError):
        pass

    # fake a single node world 
    os.environ['WORLD_SIZE'] = "1"
    os.environ['LOCAL_WORLD_SIZE'] = "1"

def get_args():
    # TODO: figure out how to assume args from the trainer hparams
    # TODO: add argument for object store instead of local saving 
    parser = ArgumentParser(description='Entrypoint for pretraining and finetuning LMs')
    parser.add_argument('-f', help='yaml file for the model to train with')
    parser.add_argument('--training_scheme', help='training scheme used (one of pretrain, finetune, or all)')
    parser.add_argument('--load_path', help='path to load checkpoint from')
    parser.add_argument('--save_folder', help='path to load checkpoint from')
    parser.add_argument('--output_type', help='type of metrics to output at the end of training scheme')

    return parser.parse_args()

def _main() -> None:
    warnings.formatwarning = _warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], '--help']

    args = get_args()

    # TODO: (ishana) make validate function to clean up args for conditions
    # TODO: allow checkpoint from object store -> load single checkpoint or directory
    # pretrain or all --> pretraining enabled, save checkpoint to save_folder
    if args.training_scheme != 'finetune': 
        root_dir = os.path.join(os.path.dirname(composer.__file__), '..')
        training_script = os.path.join(root_dir, 'examples/run_composer_trainer.py')
        proc = subprocess.Popen(['composer', training_script, '-f', f"{args.f}", '--save_folder', f"{args.save_folder}"], cwd=root_dir)
        proc.wait() # block until training is complete 
       
        # rename checkpoints to avoid rewriting if saving during finetuning
        if args.training_scheme == 'all':
            new_save_folder =os.path.join(args.save_folder, 'pretrained')
            if not os.path.exists(new_save_folder):
                os.mkdir(new_save_folder)
            for file_name in os.listdir(args.save_folder):
                old_file_path = os.path.join(args.save_folder, file_name)
                shutil.move(old_file_path, new_save_folder)

        print("PRETRAINING COMPLETE")

    if args.training_scheme != 'pretrain':
        ckpt_folder = ''
        if args.training_scheme == 'all':
            # load checkpoints from where pretraining saved them
            ckpt_folder = os.path.join(args.save_folder, 'pretrained')
            save_folder = os.path.join(args.save_folder)
        else:
            ckpt_folder = args.load_path
            save_folder = os.path.join(args.load_path, '..')

        # finetune on every pretrained checkpoint  
        for ckpt_filename in sorted(os.listdir(ckpt_folder)):
            ckpt_load_path = os.path.join(ckpt_folder, ckpt_filename)
            # skip symlinks to existing checkpoints
            if (os.path.islink(ckpt_load_path)):
                continue 

            # TODO: (ishana) PUT SEEDS BACK IN
            # finetune_tasks = ['cola', 'sst-2', 'qqp', 'qnli', 'mnli']
            # save_ckpts = [0,0,0,0,1]
            finetune_tasks = ['cola', 'mnli']
            save_ckpts = [False, True]
            # train_finetune_wrapper(finetune_tasks, save_ckpts, ckpt_load_path, ckpt_filename, save_folder)

            # finetune using the mnli checkpoint 

            # load checkpoints from where pretraining saved them
            ckpt_folder = os.path.join(save_folder, 'mnli')  

            # finetune on single mnli checkpoint 
            parent_ckpt = ckpt_filename # necessary for logging 
            ckpt_filename =  sorted(os.listdir(ckpt_folder))[0]
            ckpt_load_path = os.path.join(ckpt_folder, ckpt_filename)
            mnli_finetune_tasks = ['rte', 'mrpc', 'stsb']
            save_ckpts = [0, 0, 0]
            # delete finetuning head to reinitialize number of classes 
            train_finetune_wrapper(mnli_finetune_tasks, save_ckpts, ckpt_load_path, ckpt_filename, args.save_folder, parent_ckpt=parent_ckpt, load_ignore_keys="state/model/model.classifier*")  # type: ignore

        print('FINETUNING COMPLETE')

    if args.training_scheme != 'pretrain':
        output_metrics(output_type = args.output_type)
        return

if __name__ == '__main__':
    mp.set_start_method('spawn')
    _main()

