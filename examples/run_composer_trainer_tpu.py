# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import sys
import tempfile
import warnings
from typing import Type

from composer.loggers.logger import LogLevel
#from composer.loggers.logger_hparams import WandBLoggerHparams
from composer.trainer.trainer_hparams_tpu import TrainerTPUHparams
from composer.utils import dist
import os
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_VISIBLE_DEVICES"] = "0,1,2,3"

def train():

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    print(TrainerTPUHparams)
    hparams = TrainerTPUHparams.create(cli_args=True)  # reads cli args from sys.argv
    
            
    trainer = hparams.initialize_object()

    try:
        import wandb
    except ImportError:
        pass
    else:
        if wandb.run is not None:
            wandb.config.update(hparams.to_dict())
    
    if dist.get_global_rank() == 0:
        with tempfile.NamedTemporaryFile(mode="x+") as f:
            f.write(hparams.to_yaml())
            trainer.logger.file_artifact(LogLevel.FIT,
                                         artifact_name='bla',
                                         file_path=f.name,
                                         overwrite=True)
    import torch_xla.core.xla_model as xm
    xm.rendezvous('once')
            
    trainer.fit()

    
def _mp_fn(index):
    train()
    
if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, args=(), nprocs=8)
