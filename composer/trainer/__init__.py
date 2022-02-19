# Copyright 2021 MosaicML. All Rights Reserved.

"""Train models!

The trainer supports models with :class:`.ComposerModel` instances.
The :class:`.Trainer` is highly customizable and can
support a wide variety of workloads.

Example
--------

.. doctest::

    >>> import os
    >>> from composer import Trainer
    >>>
    >>> ### Create a trainer
    >>> trainer = Trainer(model=model,
    ...                   train_dataloader=train_dataloader,
    ...                   max_duration="1ep",
    ...                   eval_dataloader=eval_dataloader,
    ...                   optimizers=optimizer,
    ...                   schedulers=scheduler,
    ...                   device="cpu",
    ...                   validate_every_n_epochs=1,
    ...                   save_folder="checkpoints",
    ...                   save_interval="1ep")
    >>>
    >>> ### Fit and run evaluation for 1 epoch.
    >>> ### Save a checkpoint after 1 epocch as specified during trainer creation.
    >>> trainer.fit()
    >>>
    >>> ### Get the saved checkpoint folder
    >>> ### By default, the checkpoint folder is of the form runs/<timestamp>/rank_0/checkpoints
    >>> ### Alternatively, if you set the run directory environment variable as follows:
    >>> ### os.environ["COMPOSER_RUN_DIRECTORY"] = "my_run_directory", then the checkpoint path
    >>> ### will be of the form my_run_directory/rank_0/checkpoints
    >>> checkpoint_folder = trainer.checkpoint_saver.checkpoint_folder
    >>>
    >>> ### If the save_interval was in terms of epochs like above then by default,
    >>> ### checkpoint filenames are of the form "ep{EPOCH_NUMBER}.pt".
    >>> checkpoint_path = os.path.join(checkpoint_folder, "ep1.pt")
    >>>
    >>> ### Create a new trainer with the load_path argument set to the checkpoint path.
    >>> ### This will automatically load the checkpoint on trainer creation.
    >>> trainer = Trainer(model=model,
    ...                   train_dataloader=train_dataloader,
    ...                   max_duration="2ep",
    ...                   eval_dataloader=eval_dataloader,
    ...                   optimizers=optimizer,
    ...                   schedulers=scheduler,
    ...                   device="cpu",
    ...                   validate_every_n_epochs=1,
    ...                   load_path=checkpoint_path)
    >>>
    >>> ### Continue training and running evaluation where the previous trainer left off
    >>> ### until the new max_duration is reached.
    >>> ### In this case it will be one additional epoch to reach 2 epochs total.
    >>> trainer.fit()
"""
from composer.trainer import devices as devices
from composer.trainer.trainer import Trainer as Trainer
from composer.trainer.trainer_hparams import TrainerHparams as TrainerHparams

load = TrainerHparams.load

__all__ = ["Trainer"]
