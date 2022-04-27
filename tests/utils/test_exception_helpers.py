import sys

import pytest

from composer.utils import print_traceback_before_cleanup


def test_print_traceback_before_cleanup(capsys: pytest.CaptureFixture):
    with pytest.raises(RuntimeError):
        with print_traceback_before_cleanup(cleanup=lambda: print("Cleanup", file=sys.stderr)):
            raise RuntimeError("Exception!")
    captured = capsys.readouterr()
    output_lines = captured.err.split("\n")
    assert output_lines[-3] == "RuntimeError: Exception!"
    assert output_lines[-2] == "Cleanup"


def test_print_traceback_before_cleanup_repeat(capsys: pytest.CaptureFixture):
    # Raise the same exception twice, and assert that it is indeed printed twice
    exc = RuntimeError("Exception!")
    with pytest.raises(RuntimeError):
        with print_traceback_before_cleanup(cleanup=lambda: print("Cleanup", file=sys.stderr)):
            raise exc

    with pytest.raises(RuntimeError):
        with print_traceback_before_cleanup(cleanup=lambda: print("Cleanup", file=sys.stderr)):
            raise exc

    captured = capsys.readouterr()
    output_lines = captured.err.split("\n")
    num_runtime_error_lines = len(list(l for l in output_lines if l == "RuntimeError: Exception!"))
    num_cleanup_lines = len(list(l for l in output_lines if l == "Cleanup"))
    assert num_runtime_error_lines == 2
    assert num_cleanup_lines == 2
