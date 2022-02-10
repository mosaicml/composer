Methods Overview
===================

.. panels::
   :card: shadow

   ---
   .. link-button:: alibi.html
      :text: AliBi
   :badge:`NLP,badge-success`
   ++++++++++++++
   ALiBi (Attention with Linear Biases) dispenses with position embeddings for tokens in transformer-based NLP models, instead encoding position information by biasing the query-key attention scores proportionally to each token pair's distance. ALiBi yields excellent extrapolation to unseen sequence lengths compared to other position embedding schemes. We leverage this extrapolation capability by training with shorter sequence lengths, which reduces the memory and computation load.

   ---

   .. link-button:: aug_mix.html
      :text: AugMix
   :badge:`Vision,badge-primary`
   ++++++++++++++
   For each data sample, AugMix creates an augmentation chain by sampling depth image augmentations from a set (e.g. translation, shear, contrast) and applies them sequentially with randomly sampled intensity. This is repeated width times in parallel to create width different augmentation chains. The augmented images are then combined via a random convex combination to yield a single augmented image, which is in turn combined via a random convex combination sampled from a Beta(alpha, alpha) distribution with the original image.



   ---
   .. link-button:: blurpool.html
      :text: BlurPool

   ---
   .. link-button:: channels_last.html
      :text: Channels Last

   ---
   .. link-button:: col_out.html
      :text: ColOut

   ---
   .. link-button:: cut_out.html
      :text: Cutout

   ---
   .. link-button:: decoupled_weight_decay.html
      :text: Decoupled Weight Decay

   ---
   .. link-button:: factorize.html
      :text: Factorize

   ---
   .. link-button:: ghost_batchnorm.html
      :text: Ghost BatchNorm

   ---
   .. link-button:: label_smoothing.html
      :text: Label Smoothing

   ---
   .. link-button:: layer_freezing.html
      :text: Layer Freezing

   ---
   .. link-button:: mix_up.html
      :text: MixUp

   ---
   .. link-button:: progressive_resizing_vision.html
      :text: Progressive Image Resizing

   ---
   .. link-button:: rand_augment.html
      :text: RandAugment

   ---
   .. link-button:: sam.html
      :text: Sharpness Aware Minimization

   ---
   .. link-button:: scale_schedule.html
      :text: Scale Schedule

   ---
   .. link-button:: selective_backprop.html
      :text: Selective Backprop

   ---
   .. link-button:: squeeze_excite.html
      :text: Squeeze-and-Excitation

   ---
   .. link-button:: stochastic_depth_blockwise.html
      :text: Stochastic Depth (Block-Wise)

   ---
   .. link-button:: stochastic_depth_samplewise.html
      :text: Stochastic Depth (Sample-Wise)

   ---
   .. link-button:: swa.html
      :text: Stochastic Weight Averaging