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
from argparse import ArgumentParser
import os
import shutil
import signal
import subprocess
import sys
import time
# import tempfile
import warnings
import torch
import multiprocessing as mp
from typing import Optional, Type
import composer
from composer.cli.launcher import _get_free_tcp_port

# from composer.loggers.logger import LogLevel
from composer.trainer.nlp_hparams import NLPTrainerHparams

def _warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'

def prepare_output(output_type: str) -> None:
    # TODO (ishana) consolidate metrics and print to stdout depending on output_type param
    return None

def train_finetune_wrapper(finetune_tasks: list, save_ckpts: list, ckpt_load_path: str, ckpt_filename: str, save_folder: str) -> None:
    cuda_envs = init_cuda_queue(len(finetune_tasks))
    num_tasks = len(finetune_tasks)
    
    # finetuning from pretrained checkpoint
    print(f"FINETUNING ON {ckpt_filename}!")
    pool = mp.Pool(num_tasks, initializer=init_cuda_env, initargs=(cuda_envs,))
    pool.starmap(train_finetune, [(finetune_tasks[i], save_ckpts[i], ckpt_load_path, save_folder, i)
        for i in range(num_tasks)])
    # for rank in range(num_tasks):
    #     pool.apply(train_finetune, (finetune_tasks[rank], save_ckpts[rank], ckpt_load_path, save_folder, rank))
    #     print('exited apply...')
    print('exited processes...')
    cuda_envs.close() 
    cuda_envs.join_thread()
    print('finished joining thread...')
    pool.close()
    pool.join() # wait for join to proceed
    print('finished joining pool...')
    
def train_finetune(task: str, save_ckpt: bool, load_path: str, save_folder: str, gpu_id: int, set_tcp_port: Optional[bool] = False):
    # add dummy arg because arg parser removes first argument in tokenization
    sys.argv = ['dummy_arg' , '-f', f'./composer/yamls/models/glue/{task}.yaml', 
                '--load_path', load_path,
                '--device_id' , f"{gpu_id}",
                "--log_to_console", "True",
                "--progress_bar", "False",
                "--loggers", "file",
                "--filename" , "{run_name}/logs_{rank}_" + f"{task}" + ".txt"]
    # TODO  use args instead of manually entering these 
    if save_ckpt:
        sys.argv.append("--save_folder")
        task_ckpt_path = os.path.join(save_folder, f'{task}')
        sys.argv.append(task_ckpt_path)
        sys.argv.append("--save_num_checkpoints_to_keep")
        sys.argv.append("1")

        if not os.path.exists(task_ckpt_path):
            os.mkdir(task_ckpt_path)
    
    # set gpu
    print(f" --------\n SPAWNING NEW TASK: {task}\n DEVICE: {torch.cuda.current_device()}\n {sys.argv}\n --------")

    # last file in folder is the symlink to the latest ckpt file 
    ft_hparams = NLPTrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    trainer = ft_hparams.initialize_object()
    trainer.fit(duration='3ep')
    print(f"\nFINISHED TRAINING TASK {task}!!\n")
   

def init_cuda_queue(queue_size: int) -> mp.Queue:
    cuda_envs = mp.Queue(queue_size)
    cuda_envs_list = range(queue_size)
    for e in cuda_envs_list:
        cuda_envs.put(e)
    time.sleep(0.1) # Just enough to let the Queue finish

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
    parser = ArgumentParser(description='Entrypoint for pretraining and finetuning LMs')
    parser.add_argument('-f', help='yaml file for the model to train with')
    parser.add_argument('--training_scheme', help='training scheme used (one of pretrain, finetune, or all)')
    parser.add_argument('--load_path', help='path to load checkpoint from')
    parser.add_argument('--save_folder', help='path to load checkpoint from')

    return parser.parse_args()

def _main() -> None:
    warnings.formatwarning = _warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], '--help']

    args = get_args()
   
    # # if using wandb, store the config inside the wandb run
    # try:
    #     import wandb
    # except ImportError:
    #     pass
    # else:
    #     if wandb.run is not None:
    #         wandb.config.update(hparams.to_dict())

    # Only log the config once, since it should be the same on all ranks.
    # if dist.get_global_rank() == 0:
    #     with tempfile.NamedTemporaryFile(mode='x+') as f:
    #         f.write(hparams.to_yaml())
    #         trainer.logger.file_artifact(LogLevel.FIT,
    #                                      artifact_name=f'{trainer.state.run_name}/nlp_hparams.yaml',
    #                                      file_path=f.name,
    #                                      overwrite=True)

    # Print the config to the terminal and log to artifact store if on each local rank 0
    # if dist.get_local_rank() == 0:
    #     print('*' * 30)
    #     print('Config:')
    #     print(hparams.to_yaml())
    #     print('*' * 30)

    #TODO (ishana) make validate function to clean up args for conditions
    # pretrain or all --> pretraining enabled, save checkpoint to save_folder
    if args.training_scheme != 'finetune': 
        root_dir = os.path.join(os.path.dirname(composer.__file__), '..')
        training_script = os.path.join(root_dir, 'examples/run_composer_trainer.py')
        proc = subprocess.Popen(['composer', training_script, '-f', f"{args.f}", '--save_folder', f"{args.save_folder}"], 
                                        cwd=root_dir)
        proc.wait() # block until training is complete 
       
        # rename checkpoints to avoid rewriting if saving during finetuning
        if args.training_scheme == 'all':
            new_save_folder =os.path.join(args.save_folder, 'pretrained')
            if not os.path.exists(new_save_folder):
                os.mkdir(new_save_folder)
            for file_name in os.listdir(args.save_folder):
                old_file_path = os.path.join(args.save_folder, file_name)
                shutil.move(old_file_path, new_save_folder)

        print("done pretraining...")

    if args.training_scheme != 'pretrain':
        ckpt_folder = ''
        if args.training_scheme == 'all':
            # load checkpoints from where pretraining saved them
            ckpt_folder = os.path.join(args.save_folder, 'pretrained')
        else:
            ckpt_folder = args.load_path

        # finetune on every checkpoint  
        # TODO (check if checkpoints are always going to be direcotires/how to use with object store)
        for ckpt_filename in sorted(os.listdir(ckpt_folder)):
            ckpt_load_path = os.path.join(ckpt_folder, ckpt_filename)
            # skip symlinks to existing checkpoints
            if (os.path.islink(ckpt_load_path)):
                continue
            # TODO (ishana) PUT SEEDS BACK IN
            finetune_tasks = ['cola', 'sst-2', 'qqp', 'qnli']#, 'mnli']
            save_ckpts = [0,0,0,0]#, 1]
            train_finetune_wrapper(finetune_tasks, save_ckpts, ckpt_load_path, ckpt_filename, args.save_folder)

            # finetune using the mnli checkpoint 
            if args.training_scheme == 'all':
                # load checkpoints from where pretraining saved them
                ckpt_folder = os.path.join(args.save_folder, 'mnli') 
            else:
                ckpt_folder = os.path.join(args.load_path, 'mnli')

            # finetune on single mnli checkpoint  
            ckpt_filename =  os.listdir(ckpt_folder)[0]
            ckpt_load_path = os.path.join(ckpt_folder, ckpt_filename)
            mnli_finetune_tasks = ['rte', 'mrpc', 'stsb']
            save_ckpts = [0, 0, 0]
            train_finetune_wrapper(mnli_finetune_tasks, save_ckpts, ckpt_load_path, ckpt_filename, args.save_folder)

    if args.training_scheme == 'all':
        # prepare_output(output_type = output_type)
        return

if __name__ == '__main__':
    mp.set_start_method('fork')
    _main()