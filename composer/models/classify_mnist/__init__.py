# Copyright 2021 MosaicML. All Rights Reserved.

"""A simple example convolutional neural network which can be used to classify MNIST data."""
from composer.models.classify_mnist.mnist_hparams import MnistClassifierHparams as MnistClassifierHparams
from composer.models.classify_mnist.model import MNIST_Classifier as MNIST_Classifier

__all__ = ["MNIST_Classifier", "MnistClassifierHparams"]

_task = 'Image Classification'
_dataset = 'MNIST'
_name = 'SimpleConvNet'
_quality = ''
_metric = 'Accuracy'
_ttt = '?'
_hparams = 'classify_mnist_cpu.yaml'
