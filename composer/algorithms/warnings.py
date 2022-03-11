# Copyright 2021 MosaicML. All Rights Reserved.


class NoEffectWarning(Warning):
    """Warns when algorithm did not have an effect.

    Algorithms should emit this warning when its application resulted in no changes to the trainer :class:`~.State`. For
    example, if surgery algorithms that find and replace layers find no layers to replace.
    """
    pass


class NotIntendedUseWarning(Warning):
    """Warns when algorithm is being applied outside its intended setting.

    This is a catch-all generic warning to alert the user to potentially mis-intended applications. For example, when
    data augmentation that resizes the image is being applied to a very small image, or being applied to text data.
    """
    pass