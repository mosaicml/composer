Detailed Usage
===============

Optimizers & Schedulers
-----------------------


Checkpointing
-------------

Logging
-------

Numerics
--------


Time
----

.. _distributed-training:

Distributed Training
--------------------


Distributed data parallel, or DDP, is an essential technique for training large models. It is among the most common ways of parallelizing machine learning models. However, as are many things, parallelization in Python is not well supported. To get around these limitations, we offer a ``composer`` launcher to set up multiprocessing and synchronization. This script is directly analogous to the ``torch.distributed.run``, ``torchrun``, and ``deepspeed`` scripts that users may be familiar with.

The ``composer`` launcher is highly recommended for setting up DDP with the Composer trainer, and will be necessary to access advanced functionality in the future.

The ``composer`` launcher wraps your typical setup script. The wrapped script is responsible for setting up a single process's trainer.


Single-Node Example
~~~~~~~~~~~~~~~~~~~

In most cases, you will likely be training on multiple CPUs or GPUs on a single machine. Typically in this case, the only argument you need to worry about is the ``-n``/``--nproc`` argument to control how many processes should be spawned. When training with GPUs, this value should usually just be the same as the number of GPUs on your system. If using the Composer trainer, the trainer will handle the rest of the work.

For example, to train ResNet-50 efficiently with DDP on an 8-GPU system, you can use the following command:

.. code-block:: console
    
    composer -n 8 examples/run_composer_trainer.py -f composer/yamls/models/resnet50.yaml


Detailed Usage
~~~~~~~~~~~~~~

A summary of command line arguments can be obtained by typing ``composer -h`` into a command line.

.. argparse::
    :module: composer.cli.launcher
    :func: get_parser
    :prog: composer


DeepSpeed
---------


Profiling
---------


Callbacks
---------
