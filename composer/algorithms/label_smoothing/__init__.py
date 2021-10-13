# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.label_smoothing.label_smoothing import LabelSmoothing as LabelSmoothing
from composer.algorithms.label_smoothing.label_smoothing import LabelSmoothingHparams as LabelSmoothingHparams
from composer.algorithms.label_smoothing.label_smoothing import smooth_labels as smooth_labels

_name = 'Label Smoothing'
_class_name = 'LabelSmoothing'
_functional = 'smooth_labels'
_tldr = 'Smooths the labels with a uniform prior'
_attribution = '(Szegedy et al, 2015)'
_link = 'https://arxiv.org/abs/1512.00567'
_method_card = ''
