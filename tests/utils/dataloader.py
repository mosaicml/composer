# Copyright 2021 MosaicML. All Rights Reserved.

import torch.utils.data

from composer.core.types import DataLoader
from composer.datasets import DataloaderHparams, DataloaderSpec, DDPDataLoader


def get_dataloader(dataloader_spec: DataloaderSpec, dataloader_hparams: DataloaderHparams,
                   batch_size: int) -> DataLoader:
    sampler = torch.utils.data.DistributedSampler[int](
        dataloader_spec.dataset,
        drop_last=dataloader_spec.drop_last,
        shuffle=dataloader_spec.shuffle,
        rank=0,
        num_replicas=1,
    )

    dataloader = torch.utils.data.DataLoader(
        dataloader_spec.dataset,
        batch_size=batch_size,
        shuffle=False,  # set in the sampler
        num_workers=dataloader_hparams.num_workers,
        pin_memory=dataloader_hparams.pin_memory,
        drop_last=dataloader_spec.drop_last,
        sampler=sampler,
        collate_fn=dataloader_spec.collate_fn,
        worker_init_fn=dataloader_spec.worker_init_fn,
        multiprocessing_context=dataloader_spec.multiprocessing_context,
        generator=dataloader_spec.generator,
        timeout=dataloader_hparams.timeout,
        prefetch_factor=dataloader_hparams.prefetch_factor,
        persistent_workers=dataloader_hparams.persistent_workers,
    )
    return DDPDataLoader(dataloader)
