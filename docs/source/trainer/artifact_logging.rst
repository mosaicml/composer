|:hindu_temple:| Artifact Logging
=================================

Composer supports uploading artifacts, such as checkpoints and profiling traces, directly to third-party
experiment trackers (e.g. Weights & Biases) and cloud storage backends (e.g. AWS S3).

What is an artifact?
--------------------

An artifact is a file generated during training. Checkpoints, profiling traces, and log files
are the most common examples of artifacts. An artifacts must be a single, local file.
Collections of files can be combined into a single tarball, and a file can be stored in a temporary folder.

Each artifact must have a name, which is independent of the artifact's local filepath.
A remote backend that logs an artifact is responsible for storing and organizing the file by the artifact's
name. An artifact with the same name should override a previous artifact with that name. It is recommended that
artifact names include file extensions.

How are artifacts generated?
----------------------------

In Composer, individual classes, such as algorithms, callbacks, loggers, and profiler trace handlers, can generate
artifacts.

Once a artifact file has been written to disk, the class should call
:meth:`~composer.loggers.logger.Logger.file_artifact`, and the
centralized :class:`~composer.loggers.logger.Logger` will then pass the filepath and artifact name to all
LoggerDestinations, which are ultimately responsible for uploading and storing artifacts
(more on that :ref:`below <artifact_logging_uploading>`).

Below are some examples of the classes that generate artifacts and the types of artifacts they generate. For each class,
see the linked API Reference for additional documentation.

.. list-table::
    :header-rows: 1

    * - Type
      - Class Name
      - Description of Generated Artifacts
    * - Callback
      - :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`
      - Training checkpoint files
    * - Callback
      - :class:`~composer.callbacks.export_for_inference.ExportForInferenceCallback`
      - Trained models in inference formats
    * - Callback
      - :class:`~composer.callbacks.mlperf.MLPerfCallback`
      - MLPerf submission files
    * - Logger
      - :class:`~composer.loggers.file_logger.FileLogger`
      - Log files
    * - Logger
      - :class:`~composer.loggers.tensorboard_logger.TensorboardLogger`
      - Tensorboard TF Event Files
    * - Trace Handler
      - :class:`~composer.profiler.json_trace_handler.JSONTraceHandler`
      - Profiler trace files

Logging custom artifacts
------------------------

It is also possible to log custom artifacts outside of an algorithm or callback. For example:

.. testcode::

    from composer import Trainer
    from composer.loggers import LogLevel

    # Construct the trainer
    trainer = Trainer(...)

    # Log a custom artifact, such as a configuration YAML
    trainer.logger.file_artifact(
        log_level=LogLevel.FIT,
        artifact_name='hparams.yaml',
        file_path='/path/to/hparams.yaml',
    )

    # Train!
    trainer.fit()

.. _artifact_logging_uploading:

How are artifacts uploaded?
---------------------------

To store artifacts, in the ``loggers`` argument to the Trainer constructor, you must specify a
:class:`~composer.loggers.logger_destination.LoggerDestination` that implements the
:meth:`~composer.loggers.logger_destination.LoggerDestination.log_file_artifact`.

.. seealso::

    The built-in :class:`~composer.loggers.wandb_logger.WandBLogger` and
    :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`
    implement this method -- see the examples below.

The centralized Composer
:class:`~composer.loggers.logger.Logger` will invoke this method for all LoggerDestinations. If no LoggerDestination
implements this method, then artifacts will not be stored remotely.

Because LoggerDestinations can both generate and store artifacts, there is a potential for a circular dependency. As
such, it is important that any logger that generates artifacts (e.g. the Tensorboard Logger) does not also attempt
to store artifacts. Otherwise, you could run into an infinite loop!

Where can I store artifacts?
----------------------------

Composer includes two built-in LoggerDestinations to store artifacts:

*   The :class:`~composer.loggers.wandb_logger.WandBLogger` can upload Composer training artifacts
    as `W & B Artifacts <https://docs.wandb.ai/ref/python/artifact>`_, which are associated with the corresponding
    W & B project.

*   The :class:`~composer.loggers.object_store_logger.ObjectStoreLogger` can upload Composer training artifacts
    to any cloud storage backend or remote filesystem. We include integrations for AWS S3 and SFTP
    (see the :ref:`examples <artifact_logging_examples>` below), and you can write your own integration for a custom backend.


Why should I use artifact logging instead of uploading artifacts manually?
--------------------------------------------------------------------------

Artifact logging in Composer is optimized for efficiency. File uploads happen in background threads or
processes, ensuring that the training loop is not blocked due to network I/O. In other words, this feature
allows you to train the next batch while the previous checkpoint is being uploaded simultaneously.

.. _artifact_logging_examples:


Examples
--------

Below are some examples on how to configure Composer to log artifacts to various backends:

Weights & Biases Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    The :class:`~composer.loggers.wandb_logger.WandBLogger` API Reference.

.. testcode::

    from composer.loggers import WandBLogger
    from composer import Trainer

    # Configure the logger
    logger = WandBLogger(
        log_artifacts=True,  # enable artifact logging
    )

    # Define the trainer
    trainer = Trainer(
        ...,
        loggers=logger,
    )

    # Train!
    trainer.fit()

S3 Objects
^^^^^^^^^^

To log artifacts to a S3 bucket, we'll need to configure the :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`
with the :class:`~composer.utils.object_store.s3_object_store.S3ObjectStore` backend.

.. seealso::

    The :class:`~composer.loggers.object_store_logger.ObjectStoreLogger` and
    :class:`~composer.utils.object_store.s3_object_store.S3ObjectStore` API Reference.

.. testcode::

    from composer.loggers import ObjectStoreLogger
    from composer.utils.object_store import S3ObjectStore
    from composer import Trainer

    # Configure the logger
    logger = ObjectStoreLogger(
        object_store_cls=S3ObjectStore,
        object_store_kwargs={
            # Keyword arguments for the S3ObjectStore constructor.
            # See the API reference for all available arguments
            'bucket': 'my-bucket-name',
        },
    )

    # Define the trainer
    trainer = Trainer(
        ...,
        loggers=logger,
    )

    # Train!
    trainer.fit()

SFTP Filesystem
^^^^^^^^^^^^^^^

Similar to the S3 Example above, we can log artifacts to a remote SFTP filesystem.

.. seealso::

    The :class:`~composer.loggers.object_store_logger.ObjectStoreLogger` and
    :class:`~composer.utils.object_store.sftp_object_store.SFTPObjectStore` API Reference.

.. testcode::

    from composer.loggers import ObjectStoreLogger
    from composer.utils.object_store import SFTPObjectStore
    from composer import Trainer

    # Configure the logger
    logger = ObjectStoreLogger(
        object_store_cls=SFTPObjectStore,
        object_store_kwargs={
            # Keyword arguments for the SFTPObjectStore constructor.
            # See the API reference for all available arguments
            'host': 'sftp_server.example.com',
        },
    )

    # Define the trainer
    trainer = Trainer(
        ...,
        loggers=logger,
    )

    # Train!
    trainer.fit()
