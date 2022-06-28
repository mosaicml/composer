|:recycle:| Auto Grad Accum
===========================

Gradient accumulation is a technique that reduces the memory usage of a model. With gradient accumulation, the batch size is split into smaller microbatches, which are then sent through the network one at a time. The gradients from each microbatch
are then accumulated prior to a weight update. In our trainer, this behavior can be set with:

.. code:: python

    from composer import Trainer

    trainer = Trainer(
        ...,
        grad_accum=2,
    )

The setting ``grad_accum=2`` means that the batch is split into two smaller microbatches, effectively halving the total peak memory usage of the model (for more details, see :doc:. This technique is common in training language models whose memory footprint can exceed GPU memory, or in high resolution computer vision models.

Tedious to Tune
---------------

While a useful technique, it's tedious to constantly try different gradient accumulation steps to get your model to fit, and more so to adjust whenever you change the batch size, move to a different GPU such as in a colab notebook, or change how many GPUs to train across. Too high of a gradient accumulation, and each microbatch size is too small for efficient GPU computing. Too low, and your model may throw a CUDA Out of Memory (OOM) error. Finding that magical combination of gradient accumulation, batch size, number of devices, etc. in these settings can be frustrating.

Even more concerningly, some of our speed-up methods increase the memory consumption throughout training, risking frustrating OOMs deep into training.

Automatic gradient accumulation
-------------------------------

To solve these challenges, Composer now supports automatic gradient accumulation, which can be set with:

.. code:: python

    from composer import Trainer

    trainer = Trainer(
        ...,
        grad_accum='auto',
    )


With this setting, we will now catch most CUDA OOM exceptions during training, double the gradient accumulation, and attempt to retrain the batch that had failed. The trainer will start with ``grad_accum=1``, and double the setting until the memory can fit. If it reaches ``grad_accum=batch_size``, meaning that each microbatch only has 1 example, then the trainer will error out.

By setting this single parameter, most OOM errors are a thing of the past, and you can now change batch sizes, different GPUs, and scale to larger (or smaller) distributed training setups, without needing to worry about fitting the batch sizes into memory!

.. note::

    Automatic gradient accumulation also maximizes efficiency by finding the largest microbatch size that can fit into your GPU memory.

This functionality takes advantage of a unique design within our Trainer -- we ask the dataloader to return the complete batch size (also called the optimization batch size), and we handle slicing that batch into microbatches and feeding it to the model. Since we control the slicing, we can dynamically change the microbatch sizes during training.

.. note::

    This is an experimental feature, so please try it out and let us know what works, and what breaks!

For some examples of this in action, see our `blog_post <https://www.mosaicml.com/blog/farewell-oom>`_.

Caveats
-------


.. warning::

    If your model has ``BatchNorm`` layers, use with caution. Small microbatch sizes when the accumulation steps are high mean that fewer samples are used to compute these statistics. Empirically for small microbatch sizes convergence could be adversely affected (e.g. `Wu et al, 2018 <https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf>`_ and `Ying et al, 2018 <https://arxiv.org/pdf/1811.06992.pdf>`_). Using ``SyncBatchNorm`` or substituting your batch norm layer with other techniques could alleviate these issues.

Importantly, the current implementation of ``grad_accum=='auto'`` only catches OOMs that occur within the forward and backward passes during training, so a few areas that are _not_ caught yet:

* Algorithms that run the forward and backward pass outside of the trainer loop. For example, Sequence Length Warmup runs forward and backward step in order to preallocate memory, and avoid memory leaks.
* Validation can still caused OOMs if the evaluation batch size is set too high.
