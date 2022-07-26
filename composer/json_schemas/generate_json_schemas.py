# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Generate JSON Schemas for all Hparam classes."""

import os

# Dataset Hparams
from composer.datasets.ade20k_hparams import ADE20kDatasetHparams, StreamingADE20kHparams
from composer.datasets.brats_hparams import BratsDatasetHparams
from composer.datasets.c4_hparams import C4DatasetHparams, StreamingC4Hparams
from composer.datasets.cifar_hparams import CIFAR10DatasetHparams, StreamingCIFAR10Hparams
from composer.datasets.coco_hparams import COCODatasetHparams, StreamingCOCOHparams
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.evaluator_hparams import EvaluatorHparams
from composer.datasets.glue_hparams import GLUEHparams
from composer.datasets.imagenet_hparams import ImagenetDatasetHparams, StreamingImageNet1kHparams
from composer.datasets.lm_dataset_hparams import LMDatasetHparams
from composer.datasets.mnist_hparams import MNISTDatasetHparams
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
# Logger Hparams
from composer.loggers.logger_hparams_registry import ObjectStoreLoggerHparams
# Model Hparams
from composer.models.bert import BERTForClassificationHparams, BERTHparams
from composer.models.classify_mnist import MnistClassifierHparams
from composer.models.deeplabv3 import DeepLabV3Hparams
from composer.models.efficientnetb0 import EfficientNetB0Hparams
from composer.models.gpt2 import GPT2Hparams
from composer.models.model_hparams import ModelHparams
from composer.models.resnet import ResNetHparams
from composer.models.resnet_cifar import ResNetCIFARHparams
from composer.models.ssd import SSDHparams
from composer.models.timm import TimmHparams
from composer.models.unet import UnetHparams
from composer.models.vit_small_patch16 import ViTSmallPatch16Hparams
# Optimizer Hparams
from composer.optim.optimizer_hparams_registry import (AdamHparams, AdamWHparams, DecoupledAdamWHparams,
                                                       DecoupledSGDWHparams, OptimizerHparams, RAdamHparams,
                                                       RMSpropHparams, SGDHparams)
# Trainer Hparams
from composer.trainer.trainer_hparams import EvalHparams, ExperimentHparams, FitHparams, TrainerHparams
# Object Store Hparams
from composer.utils.object_store.object_store_hparams import (LibcloudObjectStoreHparams, ObjectStoreHparams,
                                                              S3ObjectStoreHparams, SFTPObjectStoreHparams)

hparam_classes = [
    ADE20kDatasetHparams, StreamingADE20kHparams, BratsDatasetHparams, C4DatasetHparams, StreamingC4Hparams,
    CIFAR10DatasetHparams, StreamingCIFAR10Hparams, COCODatasetHparams, StreamingCOCOHparams, DatasetHparams,
    DataLoaderHparams, EvaluatorHparams, GLUEHparams, ImagenetDatasetHparams, StreamingImageNet1kHparams,
    LMDatasetHparams, MNISTDatasetHparams, SyntheticHparamsMixin, ObjectStoreLoggerHparams, BERTHparams,
    BERTForClassificationHparams, MnistClassifierHparams, DeepLabV3Hparams, EfficientNetB0Hparams, GPT2Hparams,
    ResNetHparams, ResNetCIFARHparams, SSDHparams, TimmHparams, UnetHparams, ViTSmallPatch16Hparams, ModelHparams,
    AdamHparams, RAdamHparams, AdamWHparams, DecoupledAdamWHparams, SGDHparams, DecoupledSGDWHparams, RMSpropHparams,
    OptimizerHparams, TrainerHparams, FitHparams, EvalHparams, ExperimentHparams, ObjectStoreHparams,
    LibcloudObjectStoreHparams, S3ObjectStoreHparams, SFTPObjectStoreHparams
]

for hparam_class in hparam_classes:
    file_name = hparam_class.__name__.lower().replace('hparams', '')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{file_name}_hparams.json'), 'w') as f:
        hparam_class.dump_jsonschema(f=f, sort_keys=True, indent=4)
        # Lint requires ending in an empty line
        f.write('\n')
