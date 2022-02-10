# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.colout.colout import ColOut as ColOut
from composer.algorithms.colout.colout import colout_batch as colout_batch
from composer.algorithms.colout.colout import colout_image as colout_image

_name = 'ColumnOut'
_class_name = 'ColOut'
_functional = 'colout_batch'
_tldr = 'Removes columns and rows from the image for augmentation and efficiency.'
_attribution = 'MosaicML'
_link = ''
_method_card = ''
