# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous Helpers."""

import socket
from typing import Type

import torch

__all__ = ['is_model_deepspeed', 'is_notebook', 'warning_on_one_line', 'get_free_tcp_port']


def is_model_deepspeed(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`~deepspeed.DeepSpeedEngine`."""
    try:
        import deepspeed
    except ImportError:
        return False
    else:
        return isinstance(model, deepspeed.DeepSpeedEngine)


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
