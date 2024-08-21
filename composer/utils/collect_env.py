# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers to gather system information for debugging and bug reporting.

Leverages PyTorch's :mod:`torch.utils.collect_env` package to gather pertinent system information.
The following information is additionally collected to faciliate Comopser specific debug:

* Composer version
* Number of nodes
* Host processor model name
* Host processor physical core count
* Number of accelerators per node
* Accelerator model name

This package can be invoked as a standalone console script or can be invoked from within
an application to gather and generate a system environment report.

The module can be invoked by using the entrypoint alias:

.. code-block::

    $ composer_collect_env

Or manually as a standalone script:

.. code-block::

    $ python composer/utils/collect_env.py

To generate a system report from within a user application see :func:`print_env`.

A custom excepthook wrapper is also provided which extends the original :func:`sys.excepthook`
to automatically collect system information when an exception is raised.

To override the original :func:`sys.excepthook` see :func:`configure_excepthook`.

By default, the Composer custom ``excepthook`` automatically generates the environment report.
To disable automatic environment report generation, use the :func:`disable_env_report` helper
function.  Report generation can be re-enabled by using the :func:`enable_env_report` function.
"""

import functools
import json
import sys
import time
from typing import Optional, TextIO

import cpuinfo
import importlib_metadata
import psutil

from composer.utils.misc import is_notebook

__all__ = ['configure_excepthook', 'disable_env_report', 'enable_env_report', 'print_env', 'get_composer_env_dict']

# Check if PyTorch is installed
try:
    import torch.utils.collect_env as torchenv
    from torch.cuda import device_count as cuda_device_count
    from torch.cuda import get_device_name as accel_device_name
    from torch.cuda import is_available as cuda_available
    TORCH_AVAILABLE = True
except (ImportError,):
    TORCH_AVAILABLE = False

# Check if Composer is installed
try:
    import composer
    from composer.utils import dist
    COMPOSER_AVAILABLE = True
except (ImportError,):
    COMPOSER_AVAILABLE = False

# Check if we're running in a notebook
IPYTHON_AVAILABLE = is_notebook()
if IPYTHON_AVAILABLE:
    from composer.utils.import_helpers import import_object
    get_ipython = import_object('IPython:get_ipython')
    nb = get_ipython()

# Place to keep track of the original excepthook
_orig_excepthook = None

# Track if excepthook was previously registered, needed for indempotency
_EXCEPTHOOK_REGISTERED = False

# Track if environment report generation on exception is enabled, enabled by default
_ENV_EXCEPTION_REPORT = True


def get_composer_commit_hash() -> Optional[str]:
    # Use PEP-610 to get the commit hash
    # See https://packaging.python.org/en/latest/specifications/direct-url/
    # Try both package names that Composer is released under
    try:
        files = importlib_metadata.files('mosaicml')
    except importlib_metadata.PackageNotFoundError:
        try:
            files = importlib_metadata.files('composer')
        except importlib_metadata.PackageNotFoundError:
            return

    if files is None:
        return
    files = [f for f in files if str(f).endswith('direct_url.json')]
    if len(files) == 0:
        return
    f = files[0]
    direct_url = json.loads(f.read_text())
    vcs_info = direct_url.get('vcs_info', {})
    commit_id = vcs_info.get('commit_id')
    return commit_id


# Helper functions to get Composer environment information
def get_composer_version() -> str:
    """Query the Composer version."""
    return str(composer.__version__)


@functools.lru_cache(maxsize=1)
def get_cpu_model() -> str:
    """Query the host processor name."""
    cpu_info = cpuinfo.get_cpu_info()
    return str(cpu_info.get('brand_raw', 'CPU'))


def get_cpu_count() -> int:
    """Determines the number of physical host processor cores."""
    return psutil.cpu_count(logical=False)


def get_node_world_size() -> int:
    """Query the number of nodes."""
    return int(dist.get_world_size() / dist.get_local_world_size())


def get_gpu_model() -> str:
    """Query the accelerator name."""
    return accel_device_name(None) if cuda_available() else 'N/A'


def get_local_world_size() -> int:
    """Determines the number of accelerators per node."""
    return dist.get_local_world_size() if cuda_available() else 0


def get_cuda_device_count() -> int:
    """Get the number of CUDA devices on the system."""
    return cuda_device_count() if TORCH_AVAILABLE else 0


# Exception message and environment report
COMPOSER_OPEN_ISSUE_URL = 'https://github.com/mosaicml/composer/issues/new/choose'


def _exc_report(exc_type) -> None:
    """Produces exception report (exception message + environment report).

    Args:
        exc_type (Exception): Type of exception.
    """
    EXCEPTION_MSG = f'Bugs can be reported at: {COMPOSER_OPEN_ISSUE_URL}\n'

    # Don't print exception report for KeyboardInterrupt
    if not issubclass(exc_type, KeyboardInterrupt):
        if issubclass(exc_type, AssertionError):
            EXCEPTION_SEV_MSG = f'This is a Composer bug. Please submit a bug report.\n'
        elif issubclass(exc_type, RuntimeError):
            EXCEPTION_SEV_MSG = f'This could be due to user error but is most likely a Composer bug.\n'
        elif issubclass(exc_type, ValueError) or issubclass(exc_type, TypeError):
            EXCEPTION_SEV_MSG = f'This was most likely due to user error but please submit a bug report if you suspect a Composer issue.\n'
        else:
            EXCEPTION_SEV_MSG = f'If you would like support debugging, submit a bug report or reach out to us on our community channels.\n'

        print('\n-------------------\n' + EXCEPTION_SEV_MSG + EXCEPTION_MSG, file=sys.stderr)

        # Only print environment report if enabled
        global _ENV_EXCEPTION_REPORT
        if _ENV_EXCEPTION_REPORT:
            print('Please include details on how to reproduce the issue and attach the following environment report:\n')
            print_env(sys.stderr)
        else:
            print(
                "Please run the \'composer_collect_env\' utility and include your environment information with the bug report\n",
            )


def enable_env_report() -> None:
    """Enable environment report generation on exception."""
    global _ENV_EXCEPTION_REPORT
    _ENV_EXCEPTION_REPORT = True


def disable_env_report() -> None:
    """Disable environment report generation on exception."""
    global _ENV_EXCEPTION_REPORT
    _ENV_EXCEPTION_REPORT = False


# Excepthook wrapper, wraps default excepthook and prints env info
def _custom_exception_handler(type, value, tb) -> None:
    """Custom exception wrapper for sys.excepthook."""
    _exc_report(exc_type=type)
    assert _orig_excepthook
    _orig_excepthook(type, value, tb)


# Custom exception handler for IPython notebooks
def _nb_custom_exception_handler(self, type, value, tb, tb_offset=None):
    """Custom exception handler for IPython."""
    _exc_report(exc_type=type)
    self.showtraceback((type, value, tb), tb_offset=tb_offset)  # standard IPython's printout


# Public function to register excethook wrapper
def configure_excepthook() -> None:
    """Collect and print system information when :func:`sys.excepthook` is called.

    The custom exception handler causes an exception message to be printed when :func:`sys.excepthook`
    is called.  The exception message provides the user with information on the nature of the exception
    and directs the user to file GitHub issues as appropriate.

    By default, the custom exception handler also generates an environment report users can attach to
    bug reports.  Environment report generation can be optionally enabled/disabled by using the
    :func:`enable_env_report` and :func:`disable_env_report` helper functions, respectively.

    Additioanlly, the custom exceptionhook checks if the user is running from an IPython session and
    sets up the custom exception handler accordingly.

    To override the default :func:`sys.excepthook` with the custom except hook:

    .. testsetup:: composer.utils.collect_env.configure_excepthook

        import sys
        from composer.utils.collect_env import configure_excepthook
        import composer.utils.collect_env as ce

        sys.excepthook = sys.__excepthook__
        ce._EXCEPTHOOK_REGISTERED = False

    .. doctest:: composer.utils.collect_env.configure_excepthook

        >>> configure_excepthook()
        >>> sys.excepthook
        <function _custom_exception_handler at ...>
    """
    global _EXCEPTHOOK_REGISTERED
    # Needs to be indempotent across multiple trainers, don't register if we've already registered
    if not _EXCEPTHOOK_REGISTERED:
        # Custom exceptions work differntly in notebooks
        if IPYTHON_AVAILABLE:
            # Set custom handler on Exception base class to apply to all exceptions
            nb.set_custom_exc((Exception,), _nb_custom_exception_handler)
        else:
            # Save original excepthook and override
            global _orig_excepthook
            _orig_excepthook = sys.excepthook
            sys.excepthook = _custom_exception_handler

        _EXCEPTHOOK_REGISTERED = True


# Get Torch environment info
def get_torch_env() -> str:
    """Query Torch system environment via :mod:`torch.utils.collect_env`."""
    return torchenv.get_pretty_env_info()


# Composer environment information string output format
_COMPOSER_ENV_INFO_FORMAT = """
Composer Version: {composer_version}
Composer Commit Hash: {composer_commit_hash}
CPU Model: {cpu_model}
CPU Count: {cpu_count}
Number of Nodes: {num_nodes}
GPU Model: {gpu_model}
GPUs per Node: {num_gpus_per_node}
GPU Count: {num_gpus}
CUDA Device Count: {cuda_device_count}
""".strip()


# Get composer environment info as a dictionary
def get_composer_env_dict() -> dict:
    """Query Composer pertinent system information as a dict."""
    return {
        'composer_version': get_composer_version(),
        'composer_commit_hash': get_composer_commit_hash(),
        'cpu_model': get_cpu_model(),
        'cpu_count': get_cpu_count(),
        'num_nodes': get_node_world_size(),
        'gpu_model': get_gpu_model(),
        'num_gpus_per_node': get_local_world_size(),
        'num_gpus': dist.get_world_size(),
        'cuda_device_count': get_cuda_device_count(),
    }


# Get Composer environment info
def get_composer_env() -> str:
    """Query Composer pertinent system information."""
    return _COMPOSER_ENV_INFO_FORMAT.format(**get_composer_env_dict())


# Generate and print environment report
def print_env(file: Optional[TextIO] = None) -> None:
    """Generate system information report.

    Example:
    .. code-block:: python

        from composer.utils.collect_env import print_env

        print_env()

    Sample Report:

    .. code-block:: text

        ---------------------------------
        System Environment Report
        Created: 2022-04-27 00:25:33 UTC
        ---------------------------------

        PyTorch information
        -------------------
        PyTorch version: 1-91+cu111
        Is debug build: False
        CUDA used to build PyTorch: 111
        ROCM used to build PyTorch: N/A

        OS: Ubuntu 18.04.6 LTS (x86_64)
        GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
        Clang version: Could not collect
        CMake version: version 3.10.2
        Libc version: glibc-2.27

        Python version: 3.8 (64-bit runtime)
        Python platform: Linux-5.8.0-63-generic-x86_64-with-glibc2.27
        Is CUDA available: True
        CUDA runtime version: 11.1.105
        GPU models and configuration:
        GPU 0: NVIDIA GeForce RTX 3080
        GPU 1: NVIDIA GeForce RTX 3080
        GPU 2: NVIDIA GeForce RTX 3080
        GPU 3: NVIDIA GeForce RTX 3080

        Nvidia driver version: 470.57.02
        cuDNN version: Probably one of the following:
        /usr/lib/x86_64-linux-gnu/libcudnn.so.8.0.5
        /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.0.5
        /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.0.5
        /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.0.5
        /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.0.5
        /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.0.5
        /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.0.5
        HIP runtime version: N/A
        MIOpen runtime version: N/A

        Versions of relevant libraries:
        [pip3] numpy==1.22.3
        [pip3] pytorch-ranger==0.1.1
        [pip3] torch==1.9.1+cu111
        [pip3] torch-optimizer==0.1.0
        [pip3] torchmetrics==0.7.3
        [pip3] torchvision==0.10.1+cu111
        [conda] Could not collect


        Composer information
        --------------------
        Composer version: 0.8.2
        Composer commit hash: 9e14a47562def0baa414242c36954eb3083dcd46
        Host processor model name: AMD EPYC 7502 32-Core Processor
        Host processor core count: 64
        Number of nodes: 1
        Accelerator model name: NVIDIA GeForce RTX 3080
        Accelerators per node: 1
        CUDA Device Count: 4


    Args:
        file (TextIO, optional): File handle, `sys.stdout` or `sys.stderr`. Defaults to `sys.stdout`.
    """
    # Set stdout during runtime if no output file is specified
    if file is None:
        file = sys.stdout

    # Creation timestamp for report
    creation_time = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time()))
    report_hdr = f'---------------------------------\n' + \
                 f'System Environment Report        \n' + \
                 f'Created: {creation_time}\n'          + \
                 f'---------------------------------\n'

    # Torch section
    print(report_hdr, file=file)
    print('PyTorch information', file=file)
    print('-------------------', file=file)
    if TORCH_AVAILABLE:
        # Only print Torch system info if installed
        print(get_torch_env() + '\n\n', file=file)
    else:
        print('Torch not installed', file=file)

    print('Composer information', file=file)
    print('--------------------', file=file)
    if COMPOSER_AVAILABLE:
        # Only print Composer system info if installed
        print(get_composer_env() + '\n\n', file=file)
    else:
        print('Composer not installed', file=file)


# Invoke as standalone CLI script
def main() -> None:
    print('Collecting system information...')
    print_env()


if __name__ == '__main__':
    sys.exit(main())
