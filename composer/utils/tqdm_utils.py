# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers to fix :mod:`tqdm` progress bars when streamed over the network."""

# Adapted from https://github.com/tqdm/tqdm/issues/1319#issuecomment-1100951505

import os
import sys

import tqdm.std
import tqdm.utils

__all__ = ['monkeypatch_tqdm']

_disp_len = tqdm.utils.disp_len
_unicode = tqdm.utils._unicode


def _should_printer_print_new_line():
    in_kubernetes_env = os.environ.get('KUBERNETES_SERVICE_HOST') is not None
    tqdm_printer_new_line_enabled = os.environ.get('TQDM_PRINTER_NEW_LINE', '').upper() in ('1', 'TRUE')
    return in_kubernetes_env or tqdm_printer_new_line_enabled


def _new_status_printer(file):
    """Manage the printing and in-place updating of a line of characters.

    Note that if the string is longer than a line, then in-place
    updating may not work (it will print a new line at each refresh).
    """
    fp = file
    fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
    if fp in (sys.stderr, sys.stdout):
        getattr(sys.stderr, 'flush', lambda: None)()
        getattr(sys.stdout, 'flush', lambda: None)()

    def fp_write(s):
        fp.write(_unicode(s))
        fp_flush()

        if _should_printer_print_new_line():
            getattr(fp, 'write', lambda x: None)('\n')

    last_len = [0]

    def print_status(s):
        len_s = _disp_len(s)
        fp_write('\r' + s + (' ' * max(last_len[0] - len_s, 0)))
        last_len[0] = len_s

    return print_status


def monkeypatch_tqdm():
    """Monkeypatch the :meth:`tqdm.std.tqdm.status_printer` to work when being streamed over a network."""
    tqdm.std.tqdm.status_printer = staticmethod(_new_status_printer)
