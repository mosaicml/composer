# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`CopyPaste <https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf>`_ 
randomly pastes objects onto an image.

See the :doc:`Method Card </method_cards/copypaste>` for more details.
"""

from composer.algorithms.copypaste.copypaste import CopyPaste as CopyPaste
from composer.algorithms.copypaste.copypaste import copypaste_batch as copypaste_batch

__all__ = ['CopyPaste', 'copypaste_batch']
