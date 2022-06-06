# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0


class NoEffectWarning(Warning):
    """Warns when an algorithm did not have an effect.

    An algorithm should emit this warning when its application resulted in no changes to the trainer :class:`~.State`.
    For example, if surgery algorithms that find and replace layers find no layers to replace.
    """
    pass


class NotIntendedUseWarning(Warning):
    """Warns when an algorithm is being applied outside its intended setting.

    This is a catch-all generic warning to alert the user to potentially unintended applications. For example, when a
    data augmentation that resizes the image is being applied to a very small image, or being applied to text data.
    """
    pass
