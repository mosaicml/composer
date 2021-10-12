import composer.algorithms as algorithms
import composer.trainer as trainer
from composer.core.precision import Precision
from composer.datasets import SyntheticDatasetHparams
from composer.trainer.devices.device_hparams import CPUDeviceHparams


def test_load(dummy_dataset_hparams: SyntheticDatasetHparams):
    trainer_hparams = trainer.load("resnet50")
    trainer_hparams.precision = Precision.FP32
    trainer_hparams.algorithms = algorithms.load_multiple(*algorithms.list_algorithms())
    trainer_hparams.train_dataset = dummy_dataset_hparams
    trainer_hparams.val_dataset = dummy_dataset_hparams
    trainer_hparams.device = CPUDeviceHparams(1)
    my_trainer = trainer_hparams.initialize_object()
    assert isinstance(my_trainer, trainer.Trainer)
