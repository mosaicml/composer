# Copyright 2021 MosaicML. All Rights Reserved.

"""Apply Fastai's `progressive resizing <https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a
7e0735/07_sizing_and_tta.ipynb>`__ data augmentation to speed up training.

Progressive resizing initially reduces input resolution to speed up early training. Throughout training, the
downsampling factor is gradually increased, yielding larger inputs up to the original input size. A final finetuning
period is then run to finetune the model using the full-sized inputs.

See the :doc:`Method Card </method_cards/progressive_resizing>` for more details.
"""

from composer.algorithms.progressive_resizing.progressive_resizing import ProgressiveResizing as ProgressiveResizing
from composer.algorithms.progressive_resizing.progressive_resizing import resize_batch as resize_batch

__all__ = ["ProgressiveResizing", "resize_batch"]
