# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.
Specifically designed for the NLP use case, allowing pre-training and fine-tuning on 
downstream tasks to be handled within one script.

Adds a --datadir flag to conveniently set a common
data directory for both train and validation datasets.

Example that finetunes a pretrained BERT:

    >>> python examples/run_nlp_trainer.py
    -f composer/yamls/models/bert_base.yaml
    --training_scheme finetune
    --load_path ~/checkpoints
"""
import os
import signal
import sys
import time
from numpy import save
import torch
import multiprocessing as mp
from composer.cli.launcher import _get_free_tcp_port

# from composer.loggers.logger import LogLevel
from composer.trainer.nlp_hparams import NLPTrainerHparams

def prepare_output(output_type: str) -> None:
    # TODO (ishana) consolidate metrics and print to stdout depending on output_type param
    return None

def train_finetune(task: str, save_folder: str, gpu_id: int, save_ckpt: bool = False):
    # add dummy arg because arg parser removes first argument in tokenization
    sys.argv = ['dummy_arg' , '-f', f'./composer/yamls/models/glue/{task}.yaml', 
                '--load_path', os.path.join(save_folder, 'pretrained-ep0-ba1-rank0'),
                '--device_id' , f"{gpu_id}",
                "--log_to_console", "True",
                "--progress_bar", "True"]
    if save_ckpt:
        sys.argv.append("--save_folder")
        sys.argv.append(save_folder)
    
    # set gpu
    print(f" --------\n SPAWNING NEW TASK: {task}\n DEVICE: {torch.cuda.current_device()}\n {sys.argv}\n --------")
    print(f" GPU COUNT: {torch.cuda.device_count()}")
    # set launcher port 
    new_port = _get_free_tcp_port()
    os.environ['MASTER_PORT'] = f"{new_port}"    

    # last file in folder is the symlink to the latest ckpt file 
    # be careful overriding and trying to access training scheme or other args later on
    ft_hparams = NLPTrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    trainer = ft_hparams.initialize_object()
    print("------\ninit finished\n------")
    trainer.fit(duration='1ep')
    print(f"\nFINISHED TRAINING TASK {task}!!\n")

def init_cuda_env(cuda_envs) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        env=cuda_envs.get()
        torch.cuda.set_device(env)
    except (BrokenPipeError, IOError):
        pass
    # os.environ['CUDA_VISIBLE_DEVICES'] = f"{env}"
    # fake a single node world 
    os.environ['WORLD_SIZE'] = "1"

def _main() -> None:
    # finetune_tasks = ['cola', 'sst-2', 'qqp', 'qnli']#, 'mnli']
    # save_ckpt = [0, 0, 0, 0, 1]
    finetune_tasks = ['mnli']
    save_ckpt = [1]
    # move to function to initialize this
    cuda_envs = mp.Queue()
    cuda_envs_list = [0,1,2,3]
    for e in cuda_envs_list:
        cuda_envs.put(e) 

    pool = mp.Pool(1, initializer=init_cuda_env, initargs=(cuda_envs,))
    # for rank in range(4):
    #     pool.apply(train_finetune, args=(finetune_tasks[rank], './test_ckpts/', rank))

    pool.starmap(train_finetune, [(finetune_tasks[i], './test_ckpts/', i, save_ckpt) for i in range(len(finetune_tasks))])
    pool.close()
    pool.join() # wait for join to proceed

if __name__ == '__main__':
    mp.set_start_method('spawn')
    _main()