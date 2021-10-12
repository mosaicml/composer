# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.progressive_resizing.progressive_resizing import ProgressiveResizing as ProgressiveResizing
from composer.algorithms.progressive_resizing.progressive_resizing import \
    ProgressiveResizingHparams as ProgressiveResizingHparams
from composer.algorithms.progressive_resizing.progressive_resizing import resize_inputs as resize_inputs

_name = 'Progressive Resizing'
_class_name = 'ProgressiveResizing'
_functional = 'resize_inputs'
_tldr = 'Increases the input image size during training'
_attribution = 'Fast AI'
_link = 'https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb'
_method_card = ''
