import logging
from typing import Optional
import os

log = logging.getLogger(__name__)

_RUN_DIRECTORY: Optional[str] = None


def set_run_directory(run_directory: str) -> None:
    global _RUN_DIRECTORY
    os.makedirs(run_directory, exist_ok=True)
    _RUN_DIRECTORY = run_directory


def get_run_directory():
    if _RUN_DIRECTORY is None:
        raise RuntimeError("Run directory is not set. Call set_run_directory()")
    else:
        return _RUN_DIRECTORY


def get_relative_to_run_directory(path: str) -> str:
    return os.path.join(get_run_directory(), path)
