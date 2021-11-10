Distributed Training (DDP)
==========================

Distributed data parallel, or DDP, is an essential technique for training large models. It is among the most common ways of parallelizing machine learning models. However, as are many things, parallelization in Python is not well supported. To get around these limitations, we offer a ``composer`` launch script to set up multiprocessing and synchronization. This script is directly analogous to the ``torch.distributed.run``, ``torchrun``, and ``deepspeed`` scripts that users may be familiar with.

The ``composer`` script is highly recommended for setting up DDP with the Mosaic trainer, and will be necessary to access advanced functionality in the future.

The ``composer`` script wraps your typical setup script. The wrapped script is responsible for setting up a single process's trainer. 


Single-Node Example
-------------------

In most cases, you will likely be training on multiple CPUs or GPUs on a single machine. Typically in this case, the only argument you need to worry about is the ``-n``/``--nproc`` argument to control how many processes should be spawned. When training with GPUs, this value should usually just be the same as the number of GPUs on your system. If using the Mosaic trainer, the trainer will handle the rest of the work.

For example, to train ResNet-50 efficiently with DDP on an 8-GPU system, you can use the following command:

>>> composer -n 8 examples/run_mosaic_trainer.py -f composer/yamls/models/resnet50.yaml


Detailed Usage
---------------

The following information can be easily accessed by typing ``composer -h`` into a command line.

.. code-block::

    usage: composer [-h] -n NPROC [--world_size WORLD_SIZE] [--base_rank BASE_RANK] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT] [-m] training_script ...

    Utility for launching distributed jobs with composer.

    positional arguments:
    training_script       The path to the training script used to initialize a single training process. Should be followed by any command-line arguments the script should be launched with.
    training_script_args

    optional arguments:
    -h, --help            show this help message and exit
    -n NPROC, --nproc NPROC
                            The number of processes to launch on this node.
    --world_size WORLD_SIZE
                            The total number of processes to launch on allnodes. Set to -1 to default to nproc (single-node operation). Defaults to -1.
    --base_rank BASE_RANK
                            The rank of the lowest ranked process to launch on this node. Specifying a base_rank B and an nproc N will spawn processes with global ranks [B, B+1, ... B+N-1]. Defaults to 0 (single-node operation).
    --master_addr MASTER_ADDR
                            The FQDN of the node running the rank 0 worker. Defaults to 127.0.0.1 (single-node operation).
    --master_port MASTER_PORT
                            The port on the master hosting the C10d TCP store. If you are running multiple trainers on a single node, this generally needs to be unique for each one.
    -m, --module_mode     If set, run the training script as a module instead of as a script.

