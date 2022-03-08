# Copyright 2021 MosaicML. All Rights Reserved.

"""Load and save checkpoints during training."""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import textwrap
from typing import Callable, Optional, Union

import torch

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.logging.logger import Logger
from composer.core.time import Time, TimeUnit
from composer.utils import dist, reproducibility, run_directory

log = logging.getLogger(__name__)

__all__ = ["CheckpointSaver"]

_COMPOSER_STATES_FILENAME = "composer_states.pt"
_DEEPSPEED_TAG = "deepspeed"  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.


def _is_pt(path: str) -> bool:
    """Returns whether the path is a .pt file."""
    return path.endswith(".pt")


def _is_archive(path: str) -> bool:
    """Returns whether the path is a tar archive."""
    return any(path.endswith(x) for x in (".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.lzma"))


def _get_write_mode(checkpoint_name: str) -> str:
    """Get the write mode to use with :func:`tarfile.open`."""
    if checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".tar"):
        write_mode = "w"
    elif checkpoint_name.endswith(".tar.gz") or checkpoint_name.endswith(".tgz"):
        write_mode = "w:gz"
    elif checkpoint_name.endswith(".tar.bz2"):
        write_mode = "w:bz2"
    elif checkpoint_name.endswith(".tar.lzma"):
        write_mode = "w:xz"
    else:
        raise ValueError(
            textwrap.dedent(f"""\
            Checkpoint name ({checkpoint_name}) has an unsupported file extension.
            Must be one of .pt, .tar, .tgz, .tar.gz, .tar.bz2, or .tar.lzma."""))

    return write_mode


def checkpoint_periodically(interval: Union[str, int, Time]) -> Callable[[State, Event], bool]:
    """_summary_

    Args:
        interval (Union[str, int, Time]): The interval describing how often checkpoints should be
            saved. If an integer, it will be assumed to be :attr:`TimeUnit.EPOCH`.
            Otherwise, the unit must be in :attr:`TimeUnit.EPOCH` or :attr:`TimeUnit.BATCH`.

    Returns:
        Callable[[State, Event], bool]: A function that can be passed as the ``save_checkpoint``
            argument into :class:`CheckpointSaver`.
    """
    if isinstance(interval, str):
        interval = Time.from_timestring(interval)
    if isinstance(interval, int):
        interval = Time(interval, TimeUnit.EPOCH)

    if interval.unit == TimeUnit.EPOCH:
        save_event = Event.EPOCH_CHECKPOINT
    elif interval.unit == TimeUnit.BATCH:
        save_event = Event.BATCH_CHECKPOINT
    else:
        raise NotImplementedError(
            f"Unknown checkpointing interval: {interval.unit}. Must be TimeUnit.EPOCH or TimeUnit.BATCH.")

    last_checkpoint_batch = None

    def should_checkpoint(state: State, event: Event):
        nonlocal last_checkpoint_batch
        if state.get_elapsed_duration() >= 1.0:
            # if doing batch-wise checkpointing, and we saved a checkpoint at the batch_checkpoint event
            # right before the epoch_checkpoint event, do not save another checkpoint at the epoch_checkpoint
            # event if the batch count didn't increase.
            if state.timer.batch != last_checkpoint_batch:
                last_checkpoint_batch = state.timer.batch
                return True

        if event == save_event:
            if save_event == Event.EPOCH_CHECKPOINT and int(state.timer.epoch) % int(interval) == 0:
                last_checkpoint_batch = state.timer.batch
                return True
            if save_event == Event.BATCH_CHECKPOINT and int(state.timer.batch) % int(interval) == 0:
                last_checkpoint_batch = state.timer.batch
                return True

        return False

    return should_checkpoint


class CheckpointSaver(Callback):
    """Manager for saving trainer state to checkpoint files.

    Args:
        save_folder (str): Folder where checkpoints are saved.

            If an absolute path is specified, then
            that path will be used. Otherwise, the ``save_folder`` will be relative
            to the folder returned by :meth:`~composer.utils.run_directory.get_run_directory`.
            If the ``save_folder`` does not exist, it will be created.

        name_format_string (str, optional): A format string describing how to name checkpoints.
            Checkpoints will be saved approximately to ``{save_folder}/{name_format_string.format(...)}``.

            See :meth:`format_checkpoint_name` for the available format variables.

            .. note::

                *   When not using DeepSpeed, only the rank zero process will save a checkpoint file. If no
                    file extension is specified, ``.pt`` will be used.

                *   When using DeepSpeed, each rank will save a checkpoint file in tarball format. DeepSpeed
                    requires tarball format, as it saves model and optimizer states in separate files.
                    Ensure that ``{rank}`` appears within the ``name_format_string``.

                    If no file extension is specified, ``.tar`` will be used.

                *   To use compression (regardless of whether DeepSpeed is enabled), set the file extension
                    to ``.tar.gz``, ``.tgz``, ``.tar.bzip``, or ``.tar.lzma`` (depending on the desired
                    compression algorithm). Using compression will block the training loop while checkpoints are
                    being compressed. As such, we recommend saving checkpoints without compression.

            Consider the following example:

            *   The ```checkpoint_folder`` argument is set to ``"checkpoints"``
            *   The ``name_format_string`` is set to ``"ep{epoch}-ba{batch}/rank_{rank}"``
            *   DeepSpeed is not being used.
            *   The global rank of the current process is ``0``.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            Then, the checkpoint will be saved to ``"checkpoints/ep1-ba42/rank_0.pt"``.
            See :meth:`format_checkpoint_name` for the full list of available format variables.

            Default: ``"ep{epoch}-ba{batch}/rank_{rank}"``

        latest_symlink_format_string (str, optional): A format string for the name of a symlink
            (relative to ``checkpoint_folder``) that points to the last saved checkpoint.

            See :meth:`format_checkpoint_name` for the available format variables.

            To disable symlinks, set this parameter to ``None``.

            For example, setting this parameter to ````"latest/rank_{rank}"`` will create a subfolder called
            ``'latest'`` inside the ``checkpoint_folder``. Files (symlinks) inside this folder will follow the naming
            convention of ``rank_{rank}`` (``{rank}`` is a format variable corresponding to the checkpoint file's
            process rank.) Each symlink will point to the latest checkpoint file for that rank.

            Default: ``"latest/rank_{rank}"``

        overwrite (bool, optional): Whether existing checkpoints should be overridden.
            If ``False`` (the default), then the ``checkpoint_folder`` must not exist or be empty.
            (default: ``False``)

        should_checkpoint (Time | str | int | (State, Event) -> bool): A :class:`Time`, time-string, integer,
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.

            If an integer, checkpoints will be saved every n epochs.
            If :class:`Time` or a time-string, checkpoints will be saved according to this interval.

            .. seealso:: checkpoint_periodically

            If a function, then this function should take two arguments (:class:`State`, :class:`Event`).
            The latter argument will be :attr:`Event.BATCH_CHECKPOINT` or :attr:`EPOCH_CHECKPOINT`. It should return
            ``True`` if a checkpoint should be saved given the current State and Event.

        weights_only (bool): If ``True``, save only the model weights instead of the entire training state.
            This parmeter must be ``False`` when using DeepSpeed. (default: ``False``)

    Attributes:
        checkpoint_folder (str): The folder in which checkpoints are stored. If an absolute path was specified for
            ``save_folder`` upon instantiation, then that path will be used. Otherwise, this folder is relative to
            the run directory of the training run (e.g. ``{run_directory}/{save_folder}``).
            If no run directory is provided, then by default, it is of the form
            ``runs/<timestamp>/rank_<GLOBAL_RANK>/<save_folder>`` where ``timestamp``
            is the start time of the run in iso-format, ``GLOBAL_RANK`` is the global rank of the process,
            and ``save_folder`` is the save_folder argument provided upon construction.

            .. seealso:: :mod:`~composer.utils.run_directory` for details on the format of the run directory
                and how to customize it.
        saved_checkpoints (Dict[Timestamp, List[str]]): A dictionary mapping a save timestamp
            to a list of filepaths corresponding to the checkpoints saved at that time.

            .. note:: When using DeepSpeed, the index of a filepath in each list corresponds to the
                global rank of the process that wrote that file. These filepaths are valid only on
                the global rank's node. Otherwise, when not using DeepSpeed, this list will contain
                only one filepath since only rank zero saves checkpoints.
    """

    def __init__(
        self,
        save_folder: str = "checkpoints",
        name_format_string: str = "ep{epoch}-ba{batch}/rank_{rank}",
        latest_symlink_format_string: Optional[str] = "latest/rank_{rank}",
        overwrite: bool = False,
        should_checkpoint: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
        weights_only: bool = False,
    ):
        if not callable(should_checkpoint):
            should_checkpoint = checkpoint_periodically(should_checkpoint)

        self.checkpoint_folder = os.path.join(run_directory.get_run_directory(), save_folder)
        self.name_format_string = name_format_string
        self.latest_symlink_format_string = latest_symlink_format_string
        self.overwrite = overwrite

        self.should_checkpoint = should_checkpoint
        self.saved_checkpoints = {}
        self.weights_only = weights_only

    def init(self, state: State, logger: Logger) -> None:
        os.makedirs(self.checkpoint_folder, mode=0o775, exist_ok=True)
        if not self.overwrite:
            if any(x.startswith(".") for x in os.listdir(self.checkpoint_folder)):
                raise RuntimeError(
                    textwrap.dedent(f"""\
                    Checkpoint folder {self.checkpoint_folder} is not empty. When using {type(self).__name__}(overwrite=True, ...),
                    the checkpoint folder must not contain any existing checkpoints."""))

    def fit_start(self, state: State, logger: Logger) -> None:
        if state.is_model_deepspeed:
            if not _is_archive(self.name_format_string):
                raise ValueError(
                    textwrap.dedent(f"""\
                    When using deepspeed, checkpoints are stored as archives, as they contain multiple files.
                    However, `name_format_string` ({self.name_format_string}) is not an archive. To fix, the
                    `name_format_string` must end in `.tar`, `.tgz`, `.tar.gz`, `.tar.bz2, or `.tar.lzma`"""))
            if self.weights_only:
                NotImplementedError(
                    textwrap.dedent(f"""\
                    Saving checkpoints with `weights_only=True` is not currently supported when using DeepSpeed.
                    See https://github.com/mosaicml/composer/issues/685."""))

    @staticmethod
    def format_checkpoint_name(state: State, name_format_string: str):
        """Format a checkpoint name accordinate to ``name_format_string`` and the current state.

        The following format variables are available:

        +------------------------+-------------------------------------------------------+
        | Variable               | Description                                           |
        +========================+=======================================================+
        | ``{rank}``             | The global rank, as returned by                       |
        |                        | :func:`~composer.utils.dist.get_global_rank`.         |
        +------------------------+-------------------------------------------------------+
        | ``{local_rank}``       | The local rank of the process, as returned by         |
        |                        | :func:`~composer.utils.dist.get_local_rank`.          |
        +------------------------+-------------------------------------------------------+
        | ``{world_size}``       | The world size, as returned by                        |
        |                        | :func:`~composer.utils.dist.get_world_size`.          |
        +------------------------+-------------------------------------------------------+
        | ``{local_world_size}`` | The local world size, as returned by                  |
        |                        | :func:`~composer.utils.dist.get_local_world_size`.    |
        +------------------------+-------------------------------------------------------+
        | ``{node_rank}``        | The node rank, as returned by                         |
        |                        | :func:`~composer.utils.dist.node_rank`.               |
        +------------------------+-------------------------------------------------------+
        | ``{epoch}``            | The total epoch count, as returned by                 |
        |                        | :meth:`~composer.core.time.Timer.epoch`.              |
        +------------------------+-------------------------------------------------------+
        | ``{batch}``            | The total batch count, as returned by                 |
        |                        | :meth:`~composer.core.time.Timer.batch`.              |
        +------------------------+-------------------------------------------------------+
        | ``{batch_in_epoch}``   | The batch count in the current epoch, as returned by  |
        |                        | :meth:`~composer.core.time.Timer.batch_in_epoch`.     |
        +------------------------+-------------------------------------------------------+
        | ``{sample}``           | The total sample count, as returned by                |
        |                        | :meth:`~composer.core.time.Timer.sample`.             |
        +------------------------+-------------------------------------------------------+
        | ``{sample_in_epoch}``  | The sample count in the current epoch, as returned by |
        |                        | :meth:`~composer.core.time.Timer.sample_in_epoch`.    |
        +------------------------+-------------------------------------------------------+
        | ``{token}``            | The total token count, as returned by                 |
        |                        | :meth:`~composer.core.time.Timer.token`.              |
        +------------------------+-------------------------------------------------------+
        | ``{token_in_epoch}``   | The token count in the current epoch, as returned by  |
        |                        | :meth:`~composer.core.time.Timer.token_in_epoch`.     |
        +------------------------+-------------------------------------------------------+

        The resulting checkpoint file name will be approximately
        ``name_format_string.format(rank=..., local_rank=..., ...)``.

        If :attr:`name_format_string` does not specify a file extension, ``.pt`` (if not using DeepSpeed)
        or ``.tar`` (if using DeepSpeed) will be appended.

        Consider the following example:

        *   The global rank is ``0``.
        *   The current epoch count is ``1``.
        *   The current batch_count is ``42``.
        *   DeepSpeed is not being used.

        .. testsetup:: composer.callbacks.checkpoint_saver.CheckpointSaver.format_checkpoint_name

            state = State(
                train_dataloader=train_dataloader,
                max_duration='1ep',
                model=model,
                rank_zero_seed=0,
            )
            state.timer._batch.value = 42
            state.timer._epoch.value = 1
        
        .. doctest:: composer.callbacks.checkpoint_saver.CheckpointSaver.format_checkpoint_name

            >>> CheckpointSaver.format_checkpoint_name(state, "ep{epoch}-ba{batch}/rank_{rank}")
            'ep1-ba42/rank_0.pt'
        """
        checkpoint_name = name_format_string.format(
            rank=dist.get_global_rank(),
            local_rank=dist.get_local_rank(),
            world_size=dist.get_world_size(),
            local_world_size=dist.get_local_world_size(),
            node_rank=dist.get_node_rank(),
            epoch=int(state.timer.epoch),
            batch=int(state.timer.batch),
            batch_in_epoch=int(state.timer.batch_in_epoch),
            sample=int(state.timer.sample),
            sample_in_epoch=int(state.timer.sample_in_epoch),
            token=int(state.timer.token),
            token_in_epoch=int(state.timer.token_in_epoch),
        )
        if (not _is_pt(checkpoint_name)) and (not _is_archive(checkpoint_name)):
            # missing an extension
            if state.is_model_deepspeed:
                checkpoint_name += ".tar"
            else:
                checkpoint_name += ".pt"

        return checkpoint_name

    def save_checkpoint(self, state: State, checkpoint_filename: str) -> None:
        """Save the current ``state`` to ``checkpoint_filename`` (relative to :attr:`checkpoint_folder`).

        Args:
            state (State): The current State of the trainer.
            checkpoint_filename (str): The filename for the checkpoint (relative to :attr:`checkpoint_folder`).
                It will be formatted according to :meth:`format_checkpoint_name`.
        """
        state_dict = {
            'state': state.state_dict(),
            'rng': reproducibility.get_rng_state(),
        }
        if self.weights_only and not state.is_model_deepspeed:
            state_dict['state'] = {"model": state_dict['state']['model']}

        checkpoint_filename = self.format_checkpoint_name(state, checkpoint_filename)

        update_symlink = False
        with tempfile.TemporaryDirectory() as tmpdir:
            composer_states_filepath = os.path.join(tmpdir, _COMPOSER_STATES_FILENAME)
            checkpoint_filepath = os.path.join(self.checkpoint_folder, checkpoint_filename)
            if dist.get_global_rank() == 0:
                # Only rank zero saves the composer state dict
                with open(composer_states_filepath, 'xb') as f:
                    torch.save(state_dict, f)

            if state.is_model_deepspeed:
                state.deepspeed_model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)
            if _is_archive(checkpoint_filepath) and (state.is_model_deepspeed or dist.get_global_rank() == 0):
                # Either deepspeed (and every rank needs to call this),
                # or not deepspeed (but using an archive), in which case only rank zero should call this.
                os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
                write_mode = _get_write_mode(checkpoint_filepath)
                with tarfile.open(checkpoint_filepath, write_mode) as tarball:
                    # add files flat to the tarball with the specified compression
                    tarball.add(tmpdir, arcname="")
                update_symlink = True
            elif dist.get_global_rank() == 0:
                # if not an archive, then only saving the states
                # only rank zero saves the state dict
                os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
                update_symlink = True
                shutil.move(composer_states_filepath, checkpoint_filepath)

        if self.latest_symlink_format_string is not None and update_symlink:
            symlink_name = os.path.join(self.checkpoint_folder,
                                        self.format_checkpoint_name(state, self.latest_symlink_format_string))
            os.makedirs(os.path.dirname(symlink_name), exist_ok=True)
            try:
                os.remove(symlink_name)
            except FileNotFoundError:
                pass
            os.symlink(checkpoint_filepath, symlink_name)

        timestamp = state.timer.get_timestamp()
        paths = dist.all_gather_object(checkpoint_filepath if state.is_model_deepspeed else None)
        paths = list(path for path in paths if path is not None)
        self.saved_checkpoints[timestamp] = paths

        log.info('Trainer checkpoint saved to %s', checkpoint_filepath)

        # Ensure that all processes wait for the checkpoint to be saved.
        dist.barrier()

    def batch_checkpoint(self, state: State, logger: Logger):
        del logger  # unused
        if self.should_checkpoint(state, Event.BATCH_CHECKPOINT):
            return self.save_checkpoint(state, self.name_format_string)

    def epoch_checkpoint(self, state: State, logger: Logger):
        del logger  # unused
        if self.should_checkpoint(state, Event.EPOCH_CHECKPOINT):
            return self.save_checkpoint(state, self.name_format_string)
