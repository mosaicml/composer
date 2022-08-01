# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import sys
import tempfile
import warnings
from typing import Type

from composer.loggers.logger import LogLevel
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist
import torch

def train():

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
            
    trainer = hparams.initialize_object()

    try:
        import wandb
    except ImportError:
        pass
    else:
        if wandb.run is not None:
            wandb.config.update(hparams.to_dict())
        
    trainer.fit()

    
def _mp_fn(index):
    torch.set_default_tensor_type('torch.FloatTensor')
    train()
    
if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, args=(), nprocs=8)
