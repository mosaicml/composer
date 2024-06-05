from composer.models import ComposerClassifier
import torch
import itertools
from composer.trainer import Trainer
from torch.utils.data import DataLoader
from composer.utils import dist
import sys
import composer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAveragePrecision, MulticlassROC

s3_bucket = 'mosaicml-internal-checkpoints-test' #

import torch
from torch.utils.data import Dataset
from composer.models import ComposerClassifier
import contextlib
import os
import tempfile
from pathlib import Path

import torch
from composer.core import Callback, State
from composer.core.state import fsdp_state_dict_type_context, fsdp_get_optim_state_dict
from composer.loggers import Logger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.utils import (dist, format_name_with_dist_and_time, parse_uri,
                            reproducibility)

class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        return self.x[index], self.y[index]



class SimpleMLP(ComposerClassifier):
    def __init__(self, num_features: int, num_classes: int = 4, 
                 train_metrics = None,
                 val_metrics = None):
        net = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_classes, bias=False),
        )
        super().__init__(module=net, num_classes=num_classes,
                         train_metrics=train_metrics, val_metrics=val_metrics)


if __name__ == "__main__":

    (
        fsdp_state_dict_type, # ['full', 'local', 'sharded', 'no_fsdp']
        precision, # ['amp_fp16', 'amp_bf16']
        sharding_strategy, # ['FULL_SHARD'], 'SHARD_GRAD_OP']
    ) = sys.argv[1:]

    num_features=32
    num_classes=8

    fsdp_config= {
            'min_params': 1,
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': sharding_strategy,
        } if fsdp_state_dict_type != 'no_fsdp' else None
    
    folder_name = f'{sharding_strategy.lower()}_{fsdp_state_dict_type}_{precision}' if fsdp_state_dict_type != 'no_fsdp' else f'no_fsdp_{precision}'
    
    model = SimpleMLP(num_features=num_features,
                      num_classes=num_classes,
                      train_metrics=MetricCollection([MulticlassAccuracy(num_classes=num_classes),
                                                      MulticlassAveragePrecision(num_classes=num_classes),
                                                      MulticlassROC(num_classes=num_classes)]),
                      val_metrics=MetricCollection([MulticlassAccuracy(num_classes=num_classes), MulticlassAveragePrecision(num_classes=num_classes), MulticlassROC(num_classes=num_classes)]),)
    dataset = RandomClassificationDataset(shape=(num_features, ), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=32)
    optim = torch.optim.Adam(params=model.parameters())
    #rank = 0 if fsdp_state_dict_type == 'full' else '{rank}'
    trainer = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config=fsdp_config,
        save_folder=f's3://{s3_bucket}/read_only/backwards_compatibility/{composer.__version__}/{folder_name}',
        max_duration='2ba',
        save_interval='2ba',
        save_filename='ba{batch}_rank{rank}.pt',
        save_overwrite=False,
        precision=precision,
        #load_path=f'./foo/{sharding_strategy.lower()}_{fsdp_state_dict_type}_{precision}/rank{rank}.pt',
        progress_bar=False,
        log_to_console=True,
        autoresume=False,
        save_num_checkpoints_to_keep=0,
    )
    trainer.fit()
    trainer.close()
