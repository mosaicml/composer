# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous Helpers."""

import socket
from contextlib import contextmanager
from typing import Type

import torch
from torch.nn.parallel import DistributedDataParallel

__all__ = [
    'is_model_deepspeed', 'is_model_fsdp', 'is_notebook', 'warning_on_one_line', 'get_free_tcp_port', 'model_eval_mode'
]


def is_model_deepspeed(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`~deepspeed.DeepSpeedEngine`."""
    try:
        import deepspeed
    except ImportError:
        return False
    else:
        return isinstance(model, deepspeed.DeepSpeedEngine)


def is_model_ddp(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`.DistributedDataParallel`."""
    return isinstance(model, DistributedDataParallel)


def is_model_fsdp(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`.FullyShardedDataParallel`."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        is_fsdp = False
        # Check if model is wrapped with FSDP
        for _, obj in model.named_children():
            if isinstance(obj, FSDP):
                is_fsdp = True
        return is_fsdp
    except ImportError:
        return False


def is_notebook():
    """Whether Composer is running in a IPython/Jupyter Notebook."""
    try:
        __IPYTHON__  #type: ignore
        return True
    except NameError:
        return False


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    """Force Python warnings to consolidate into one line."""
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def get_free_tcp_port() -> int:
    """Get free socket port to use as MASTER_PORT."""
    # from https://www.programcreek.com/python/?CodeExample=get+free+port
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


@contextmanager
def model_eval_mode(model: torch.nn.Module):
    """Set model.eval() for context duration, restoring model status at end."""
    is_training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(mode=is_training)
