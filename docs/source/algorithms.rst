composer.algorithms
===================

.. _master-algo-list:

.. currentmodule:: composer.algorithms

We describe programmatic modifications to the model or training process as "algorithms." Examples include :py:class:`smoothing the labels <composer.algorithms.label_smoothing.LabelSmoothing>` and adding :py:class:`Squeeze-and-Excitation <composer.algorithms.squeeze_excite.SqueezeExcite>` blocks, among many others.

Algorithms can be used in two ways:

* Using :py:class:`~composer.Algorithm` objects. These objects provide callbacks to be run in the training loop.
* Using algorithm-specific functions and classes, such as :py:func:`smooth_labels <composer.algorithms.label_smoothing.smooth_labels>` or :py:class:`~composer.algorithms.squeeze_excite.SqueezeExcite2d`.

The former are the easier to compose together, since they all have the same public interface and work automatically with the Composer :py:class:`~composer.trainer.Trainer`. The latter are easier to integrate piecemeal into an existing codebase.

See :py:class:`~composer.Algorithm` for more information.

The following algorithms are available in Composer:

.. autosummary::
    :nosignatures:

    ~alibi.Alibi
    ~augmix.AugMix
    ~blurpool.BlurPool
    ~channels_last.ChannelsLast
    ~colout.ColOut
    ~cutout.CutOut
    ~composer.algorithms.factorize.Factorize
    ~ghost_batchnorm.GhostBatchNorm
    ~label_smoothing.LabelSmoothing
    ~layer_freezing.LayerFreezing
    ~mixup.MixUp
    ~progressive_resizing.ProgressiveResizing
    ~randaugment.RandAugment
    ~sam.SAM
    ~scale_schedule.ScaleSchedule
    ~selective_backprop.SelectiveBackprop
    ~seq_length_warmup.SeqLengthWarmup
    ~squeeze_excite.SqueezeExcite
    ~stochastic_depth.StochasticDepth
    ~swa.SWA

Alibi
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.alibi.Alibi
.. autoclass:: composer.algorithms.alibi.AlibiHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.alibi.apply_alibi


Augmix
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.augmix.AugMix
.. autoclass:: composer.algorithms.augmix.AugMixHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.augmix.augmix_image


BlurPool
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.blurpool.BlurPool
.. autoclass:: composer.algorithms.blurpool.BlurPoolHparams


Standalone
^^^^^^^^^^

.. autoclass:: composer.algorithms.blurpool.BlurConv2d
.. autoclass:: composer.algorithms.blurpool.BlurMaxPool2d
.. autoclass:: composer.algorithms.blurpool.BlurPool2d
.. autofunction:: composer.algorithms.blurpool.blur_2d
.. autofunction:: composer.algorithms.blurpool.apply_blurpool


Channels Last
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.channels_last.ChannelsLast
.. autoclass:: composer.algorithms.channels_last.ChannelsLastHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.channels_last.apply_channels_last


ColOut
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.colout.ColOut
.. autoclass:: composer.algorithms.colout.ColOutHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.colout.colout_image
.. autofunction:: composer.algorithms.colout.colout_batch


CutOut
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.cutout.CutOut
.. autoclass:: composer.algorithms.cutout.CutOutHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.cutout.cutout


Ghost Batch Normalization
-------------------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.ghost_batchnorm.GhostBatchNorm
.. autoclass:: composer.algorithms.ghost_batchnorm.GhostBatchNormHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.ghost_batchnorm.apply_ghost_batchnorm


Factorize
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.factorize.Factorize
.. autoclass:: composer.algorithms.factorize.FactorizeHparams


Standalone
^^^^^^^^^^

.. autoclass:: composer.algorithms.factorize.FactorizedConv2d
    :members:

.. autoclass:: composer.algorithms.factorize.FactorizedLinear
    :members:

.. autoclass:: composer.algorithms.factorize.LowRankSolution
.. autofunction:: composer.algorithms.factorize.factorize_conv2d_modules
.. autofunction:: composer.algorithms.factorize.factorize_linear_modules
.. autofunction:: composer.algorithms.factorize.factorize_matrix
.. autofunction:: composer.algorithms.factorize.factorize_conv2d


Label Smoothing
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.label_smoothing.LabelSmoothing
.. autoclass:: composer.algorithms.label_smoothing.LabelSmoothingHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.label_smoothing.smooth_labels


Layer Freezing
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.layer_freezing.LayerFreezing
.. autoclass:: composer.algorithms.layer_freezing.LayerFreezingHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.layer_freezing.freeze_layers


MixUp
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.mixup.MixUp
.. autoclass:: composer.algorithms.mixup.MixUpHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.mixup.mixup_batch


Progressive Resizing
--------------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.progressive_resizing.ProgressiveResizing
.. autoclass:: composer.algorithms.progressive_resizing.ProgressiveResizingHparams

Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.progressive_resizing.resize_inputs


RandAugment
---------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.randaugment.RandAugment
.. autoclass:: composer.algorithms.randaugment.RandAugmentHparams


Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.randaugment.randaugment

Sequence Length Warmup
----------------------

Algorithm
^^^^^^^^^

.. automodule:: composer.algorithms.seq_length_warmup
    :show-inheritance:

    .. autoclass:: composer.algorithms.seq_length_warmup.SeqLengthWarmup
    .. autoclass:: composer.algorithms.seq_length_warmup.SeqLengthWarmupHparams

Standalone
^^^^^^^^^^

.. automodule:: composer.algorithms.seq_length_warmup
   :show-inheritance:
   :noindex:

   .. autofunction:: composer.algorithms.seq_length_warmup.apply_seq_length_warmup


Sharpness-Aware Minimization
----------------------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.sam.SAM
.. autoclass:: composer.algorithms.sam.SAMHparams


Scaling the Learning Rate Schedule
----------------------------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.scale_schedule.ScaleSchedule
.. autoclass:: composer.algorithms.scale_schedule.ScaleScheduleHparams


Standalone
^^^^^^^^^^

.. autofunction:: composer.algorithms.scale_schedule.scale_scheduler


Selective Backpropagation
-------------------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.selective_backprop.SelectiveBackprop
.. autoclass:: composer.algorithms.selective_backprop.SelectiveBackpropHparams


Squeeze-and-Excitation
----------------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.squeeze_excite.SqueezeExcite
.. autoclass:: composer.algorithms.squeeze_excite.SqueezeExciteHparams


Standalone
^^^^^^^^^^

.. autoclass:: composer.algorithms.squeeze_excite.SqueezeExcite2d
.. autoclass:: composer.algorithms.squeeze_excite.SqueezeExciteConv2d
.. autofunction:: composer.algorithms.squeeze_excite.apply_se


Stochastic Depth
----------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.stochastic_depth.StochasticDepth
.. autoclass:: composer.algorithms.stochastic_depth.StochasticDepthHparams


Standalone
^^^^^^^^^^

.. autoclass:: composer.algorithms.stochastic_depth.StochasticBottleneck
.. autofunction:: composer.algorithms.stochastic_depth.apply_stochastic_depth


Stochastic Weight Averaging
---------------------------

Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.swa.SWA
.. autoclass:: composer.algorithms.swa.SWAHparams
