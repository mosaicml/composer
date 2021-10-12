composer.algorithms
===================


Alibi
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.alibi
    :show-inheritance:

    .. autoclass:: composer.algorithms.alibi.Alibi
    .. autoclass:: composer.algorithms.alibi.AlibiHparams


Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.alibi
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.alibi.apply_alibi


Augmix
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.augmix
    :show-inheritance:

    .. autoclass:: composer.algorithms.augmix.AugMix
    .. autoclass:: composer.algorithms.augmix.AugMixHparams


Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.augmix
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.augmix.augment_and_mix


BlurPool
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.blurpool
    :show-inheritance:

    .. autoclass:: composer.algorithms.blurpool.BlurPool
    .. autoclass:: composer.algorithms.blurpool.BlurPoolHparams


Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.blurpool
   :show-inheritance:
   :noindex:

   .. autoclass:: composer.algorithms.blurpool.BlurConv2d
   .. autoclass:: composer.algorithms.blurpool.BlurMaxPool2d
   .. autoclass:: composer.algorithms.blurpool.BlurPool2d
   .. autofunction:: composer.algorithms.blurpool.blur_2d
   .. autofunction:: composer.algorithms.blurpool.apply_blurpool


Channels Last
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.channels_last
    :show-inheritance:

    .. autoclass:: composer.algorithms.channels_last.ChannelsLast
    .. autoclass:: composer.algorithms.channels_last.ChannelsLastHparams


ColOut
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.colout
    :show-inheritance:

    .. autoclass:: composer.algorithms.colout.ColOut
    .. autoclass:: composer.algorithms.colout.ColOutHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.colout
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.colout.colout

CutOut
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.cutout
    :show-inheritance:

    .. autoclass:: composer.algorithms.cutout.CutOut
    .. autoclass:: composer.algorithms.cutout.CutOutHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.cutout
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.cutout.cutout


Ghost Batch Normalization
-------------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.ghost_batchnorm
    :show-inheritance:

    .. autoclass:: composer.algorithms.ghost_batchnorm.GhostBatchNorm
    .. autoclass:: composer.algorithms.ghost_batchnorm.GhostBatchNormHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.cutout
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.ghost_batchnorm.apply_ghost_batchnorm


Label Smoothing
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.label_smoothing
    :show-inheritance:

    .. autoclass:: composer.algorithms.label_smoothing.LabelSmoothing
    .. autoclass:: composer.algorithms.label_smoothing.LabelSmoothingHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.label_smoothing
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.label_smoothing.smooth_labels


Layer Freezing
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.layer_freezing
    :show-inheritance:

    .. autoclass:: composer.algorithms.layer_freezing.LayerFreezing
    .. autoclass:: composer.algorithms.layer_freezing.LayerFreezingHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.layer_freezing
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.layer_freezing.freeze_layers


MixUp
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.mixup
    :show-inheritance:

    .. autoclass:: composer.algorithms.mixup.MixUp
    .. autoclass:: composer.algorithms.mixup.MixUpHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.mixup
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.mixup.mixup_batch


Progressive Resizing
--------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.progressive_resizing
    :show-inheritance:

    .. autoclass:: composer.algorithms.progressive_resizing.ProgressiveResizing
    .. autoclass:: composer.algorithms.progressive_resizing.ProgressiveResizingHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.progressive_resizing
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.progressive_resizing.resize_inputs


RandAugment
---------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.randaugment
    :show-inheritance:

    .. autoclass:: composer.algorithms.randaugment.RandAugment
    .. autoclass:: composer.algorithms.randaugment.RandAugmentHparams


Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.randaugment
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.randaugment.randaugment



Sharpness-Aware Minimization
----------------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.sam
    :show-inheritance:

    .. autoclass:: composer.algorithms.sam.SAM
    .. autoclass:: composer.algorithms.sam.SAMHparams



Scaling the Learning Rate Schedule
----------------------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.scale_schedule
    :show-inheritance:

    .. autoclass:: composer.algorithms.scale_schedule.ScaleSchedule
    .. autoclass:: composer.algorithms.scale_schedule.ScaleScheduleHparams


Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.randaugment
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.scale_schedule.scale_scheduler


Selective Backpropagation
-------------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.selective_backprop
    :show-inheritance:

    .. autoclass:: composer.algorithms.selective_backprop.SelectiveBackprop
    .. autoclass:: composer.algorithms.selective_backprop.SelectiveBackpropHparams


Squeeze-and-Excitation
----------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.squeeze_excite
    :show-inheritance:

    .. autoclass:: composer.algorithms.squeeze_excite.SqueezeExcite
    .. autoclass:: composer.algorithms.squeeze_excite.SqueezeExciteHparams


Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.blurpool
   :show-inheritance:
   :noindex:

   .. autoclass:: composer.algorithms.squeeze_excite.SqueezeExcite2d
   .. autoclass:: composer.algorithms.squeeze_excite.SqueezeExciteConv2d
   .. autofunction:: composer.algorithms.squeeze_excite.apply_se


Stochastic Depth
----------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.stochastic_depth
    :show-inheritance:

    .. autoclass:: composer.algorithms.stochastic_depth.StochasticDepth
    .. autoclass:: composer.algorithms.stochastic_depth.StochasticDepthHparams


Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.blurpool
   :show-inheritance:
   :noindex:

   .. autoclass:: composer.algorithms.stochastic_depth.StochasticBottleneck
   .. autofunction:: composer.algorithms.stochastic_depth.apply_stochastic_depth


Stochastic Weight Averaging
---------------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.swa
    :show-inheritance:

    .. autoclass:: composer.algorithms.swa.SWA
    .. autoclass:: composer.algorithms.swa.SWAHparams
