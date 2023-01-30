# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A simple example convolutional neural network which can be used to classify MNIST data."""
from composer.models.classify_mnist.model import mnist_model as mnist_model

__all__ = ['mnist_model']

_task = 'Image Classification'
_dataset = 'MNIST'
_name = 'SimpleConvNet'
_quality = ''
_metric = 'Accuracy'
_ttt = '?'
_hparams = 'classify_mnist_cpu.yaml'
