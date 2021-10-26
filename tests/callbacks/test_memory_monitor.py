# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import MagicMock

import pytest

from composer.callbacks import MemoryMonitorHparams
from composer.datasets.synthetic import SyntheticDatasetHparams
from composer.trainer import TrainerHparams
from composer.trainer.devices.device_gpu import DeviceGPU
from torch.cuda import device_count


def _do_trainer_fit(mosaic_trainer_hparams: TrainerHparams,
    aggregate_device_stats=False
):
    memory_monitor_hparams = MemoryMonitorHparams(aggregate_device_stats)
    mosaic_trainer_hparams.callbacks.append(memory_monitor_hparams)

    mosaic_trainer_hparams.ddp.fork_rank_0 = False
    mosaic_trainer_hparams.max_epochs = 20

    mosaic_trainer_hparams.total_batch_size = 50

    trainer = mosaic_trainer_hparams.initialize_object()

    trainer.device = DeviceGPU(True, 1)

    log_destination = MagicMock()
    log_destination.will_log.return_value = True
    trainer.logger.backends = [log_destination]
    trainer.fit()

    assert isinstance(mosaic_trainer_hparams.train_dataset, SyntheticDatasetHparams)

    return log_destination


@pytest.mark.timeout(60)
@pytest.mark.run_long
def test_memory_monitor(mosaic_trainer_hparams: TrainerHparams):
    n_cuda_devices = device_count()
    if n_cuda_devices > 0:
        log_destination = _do_trainer_fit(mosaic_trainer_hparams)
    
        memory_monitor_logged = False
        for log_call in log_destination.log_metric.mock_calls:
            metrics = log_call[1][3]
            if "memory/alloc_requests" in metrics:
                memory_monitor_logged = True
        assert memory_monitor_logged