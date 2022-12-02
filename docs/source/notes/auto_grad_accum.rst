|:recycle:| Automatic Microbatching
===========================

Gradient accumulation is a technique that reduces the memory usage of a model. With gradient accumulation, the batch size is split into smaller microbatches, which are then sent through the network one at a time. The gradients from each microbatch
are then accumulated prior to a weight update. In our trainer, this behavior can be set with:

.. code:: python

    from composer import Trainer

    trainer = Trainer(
        ...,
        device_train_microbatch_size=1024,
    )

The setting ``device_train_microbatch_size=1024`` means that the batch is split into smaller microbatches of size up to 1024, If the batch size was 2048, this would effectively half the total peak memory usage of the model. This technique is common in training language models whose memory footprint can exceed GPU memory, or in high resolution computer vision models.

Tedious to Tune
---------------

While a useful technique, it's tedious to constantly try different device microbatch sizes to get your model to fit, and more so to adjust whenever move to a different GPU such as in a colab notebook or change how many GPUs to train across. Too low of a microbatch size, and each microbatch size is too small for efficient GPU computing. Too high, and your model may throw a CUDA Out of Memory (OOM) error. Finding that magical combination of device microbatch size, batch size, number of devices, etc. in these settings can be frustrating.

Even more concerningly, some of our speed-up methods increase the memory consumption throughout training, risking frustrating OOMs deep into training.

Automatic microbatching
-------------------------------

To solve these challenges, Composer now supports automatic microbatching, which can be set with:

.. code:: python

    from composer import Trainer

    trainer = Trainer(
        ...,
        device_train_microbatch_size='auto',
    )


With this setting, we will now catch most CUDA OOM exceptions during training, half the device microbatch size, and attempt to retrain the batch that had failed. The trainer will start with ``device_train_microbatch_size=device_batch_size``, and half the setting until the memory can fit. If it reaches ``device_train_microbatch_size=1``, meaning that each microbatch only has 1 example, then the trainer will error out.

By setting this single parameter, most OOM errors are a thing of the past, and you can now change batch sizes, different GPUs, and scale to larger (or smaller) distributed training setups, without needing to worry about fitting the batch sizes into memory!

.. note::

    Automatic gradient accumulation also maximizes efficiency by finding the largest microbatch size that can fit into your GPU memory.

This functionality takes advantage of a unique design within our Trainer -- we ask the dataloader to return the complete batch size (also called the optimization batch size), and we handle slicing that batch into microbatches and feeding it to the model. Since we control the slicing, we can dynamically change the microbatch sizes during training.

As a bonus feature, we also support automatic microbatching for evaluation when using the ``device_eval_microbatch_size='auto'`` setting, so the eval batch size doesn't need to be tuned either when changing GPUs or batch sizes.

.. note::

    This is an experimental feature, so please try it out and let us know what works, and what breaks!

For some examples of this in action, see our `blog_post <https://www.mosaicml.com/blog/farewell-oom>`_.

Caveats
-------


.. warning::

    If your model has ``BatchNorm`` layers, use with caution. Small microbatch sizes when the accumulation steps are high mean that fewer samples are used to compute these statistics. Empirically for small microbatch sizes convergence could be adversely affected (e.g. `Wu et al, 2018 <https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf>`_ and `Ying et al, 2018 <https://arxiv.org/pdf/1811.06992.pdf>`_). Using ``SyncBatchNorm`` or substituting your batch norm layer with other techniques could alleviate these issues.

Importantly, the current implementation of ``device_train_microbatch_size=='auto'`` only catches OOMs that occur within the forward and backward passes during training, so a few areas that are _not_ caught yet:

* Algorithms that run the forward and backward pass outside of the trainer loop.
* Callbacks which allocate memory to device that could cause an OOM outside the trainer loop.
