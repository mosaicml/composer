|:hindu_temple:| File Uploading
===============================

Composer supports uploading files, such as checkpoints and profiling traces, directly to third-party
experiment trackers (e.g. Weights & Biases) and cloud storage backends (e.g. AWS S3).

What files might I want to upload?
----------------------------------

Checkpoints, profiling traces, and log files generated during training
are the most common examples. Each file to upload must be a single, local file.
Collections of files can be combined into a single tarball, and a file can be stored in a temporary folder.

Each remote file must have a name, which is independent of the file's local filepath.
A remote backend is responsible for storing and organizing the file by the file's
name. A remote file with the same name should override a previous remote file with that name. It is recommended that
remote file names include file extensions.

How are remote files generated?
-------------------------------

In Composer, individual classes, such as algorithms, callbacks, loggers, and profiler trace handlers, can generate
files to be uploaded.

Once a file has been written to disk, the class should call
:meth:`~composer.loggers.logger.Logger.upload_file`, and the
centralized :class:`~composer.loggers.logger.Logger` will then pass the filepath and remote file name to all
LoggerDestinations, which are ultimately responsible for uploading and storing remote files
(more on that :ref:`below <file_uploading>`).

Below are some examples of the classes that generate files that might be uploaded and the types of files they generate. For each class,
see the linked API Reference for additional documentation.

.. list-table::
    :header-rows: 1

    * - Type
      - Class Name
      - Description of Generated Files
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

Saving custom files
-------------------

It is also possible to upload custom files outside of an algorithm or callback. For example:

.. testcode::

    from composer import Trainer

    # Construct the trainer
    trainer = Trainer(...)

    # Upload a custom file, such as a configuration YAML
    trainer.logger.upload_file(
        remote_file_name='hparams.yaml',
        file_path='/path/to/hparams.yaml',
    )

    # Train!
    trainer.fit()

.. _file_uploading:

How are files uploaded?
-----------------------

To store files remotely, in the ``loggers`` argument to the Trainer constructor, you must specify a
:class:`~composer.loggers.logger_destination.LoggerDestination` that implements the
:meth:`~composer.loggers.logger_destination.LoggerDestination.upload_file`.

.. seealso::

    The built-in :class:`~composer.loggers.wandb_logger.WandBLogger` and
    :class:`~composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader`
    implement this method -- see the examples below.

The centralized Composer
:class:`~composer.loggers.logger.Logger` will invoke this method for all LoggerDestinations. If no LoggerDestination
implements this method, then files will not be stored remotely.

Because LoggerDestinations can both generate and store files, there is a potential for a circular dependency. As
such, it is important that any logger that generates files that are going to be uploaded (e.g. the Tensorboard Logger) does not also attempt
to upload them. Otherwise, you could run into an infinite loop!

Where can I remotely store files?
---------------------------------

Composer includes two built-in LoggerDestinations to store artifacts:

*   The :class:`~composer.loggers.wandb_logger.WandBLogger` can upload Composer training files
    as `W & B Artifacts <https://docs.wandb.ai/ref/python/artifact>`_, which are associated with the corresponding
    W & B project.

*   The :class:`~composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader` can upload Composer training files
    to any cloud storage backend or remote filesystem. We include integrations for AWS S3 and SFTP
    (see the :ref:`examples <file_uploading_examples>` below), and you can write your own integration for a custom backend.


Why should I use built in file uploading instead of uploading files manually?
-----------------------------------------------------------------------------

File uploading in Composer is optimized for efficiency. File uploads happen in background threads or
processes, ensuring that the training loop is not blocked due to network I/O. In other words, this feature
allows you to train the next batch while the previous checkpoint is being uploaded simultaneously.

.. _file_uploading_examples:


Examples
--------

Below are some examples on how to configure Composer to upload files to various backends:

Weights & Biases Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    The :class:`~composer.loggers.wandb_logger.WandBLogger` API Reference.

.. testcode::
    :skipif: not _WANDB_INSTALLED

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

To upload files to an S3 bucket, we'll need to configure the :class:`~composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader`
with the :class:`~composer.utils.object_store.s3_object_store.S3ObjectStore` backend.

.. seealso::

    The :class:`~composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader` and
    :class:`~composer.utils.object_store.s3_object_store.S3ObjectStore` API Reference.

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    from composer.loggers import RemoteUploaderDownloader
    from composer.utils.object_store import S3ObjectStore
    from composer import Trainer

    # Configure the logger
    logger = RemoteUploaderDownloader(
        bucket_uri="s3://my-bucket-name",
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

Similar to the S3 Example above, we can upload files to a remote SFTP filesystem.

.. seealso::

    The :class:`~composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader` and
    :class:`~composer.utils.object_store.sftp_object_store.SFTPObjectStore` API Reference.

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    from composer.loggers import RemoteUploaderDownloader
    from composer.utils.object_store import SFTPObjectStore
    from composer import Trainer

    # Configure the logger
    logger = RemoteUploaderDownloader(
        bucket_uri="sftp://sftp_server.example.com",
    )

    # Define the trainer
    trainer = Trainer(
        ...,
        loggers=logger,
    )

    # Train!
    trainer.fit()
