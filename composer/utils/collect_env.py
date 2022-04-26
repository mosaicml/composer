# Copyright 2021 MosaicML. All Rights Reserved.

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

    composer_collect_env

Or manually as a standalone script:

.. code-block::

    python composer/utils/collect_env.py

To generate a system report from within a user application see :func:`print_env`.

A custom excepthook wrapper is also provided which extends the default :func:`sys.excepthook`
to automatically collect system information when an exception is raised.

To override the default :func:`sys.excepthook` see :func:`configure_excepthook`.

To restore the default :func:`sys.excepthook` see :func:`restore_excepthook`.
"""

import sys
import time
from typing import NamedTuple

import cpuinfo
import psutil

from composer.utils.import_helpers import import_object

__all__ = ["configure_excepthook", "print_env", "restore_excepthook"]

# Check if PyTorch is installed
try:
    import torch.utils.collect_env as torchenv
    from torch.cuda import get_device_name as accel_device_name
    from torch.cuda import is_available as cuda_available
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# Check if Composer is installed
try:
    import composer
    from composer.utils import dist
    COMPOSER_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    COMPOSER_AVAILABLE = False

# Check if we're running in a notebook
try:
    __IPYTHON__  #type: ignore
    IPYTHON_AVAILABLE = True
except (NameError,):
    IPYTHON_AVAILABLE = False

if IPYTHON_AVAILABLE:
    get_ipython = import_object('IPython:get_ipython')
    nb = get_ipython()

COMPOSER_OPEN_ISSUE_URL = "https://github.com/mosaicml/composer/issues/new/choose"
EXCEPTION_MSG = f"If you believe this exception was raised due to a Composer bug, " + \
                f"file a bug report at: {COMPOSER_OPEN_ISSUE_URL}\n" + \
                f"Please include the following environment report:\n"


# Same convention as Torch collect_env, create a namedtuple to track collected fields
class ComposerEnv(NamedTuple):
    composer_version: str
    number_of_nodes: int
    host_processor_model_name: str
    host_processor_core_count: int
    accelerators_per_node: int
    accelerator_model_name: str


# Helper functions to get Composer environment information
def get_composer_version() -> str:
    """Query the Composer version."""
    return str(composer.__version__)


def get_num_nodes() -> int:
    """Query the number of nodes."""
    return int(dist.get_world_size() / dist.get_local_world_size())


def get_host_processor_name() -> str:
    """Query the host processor name."""
    cpu_info = cpuinfo.get_cpu_info()
    return str(cpu_info.get('brand_raw', "CPU"))


def get_host_processor_cores() -> int:
    """Determine the number of physical host processor cores."""
    return psutil.cpu_count(logical=False)


def get_accel_per_node() -> int:
    """Determine the number of accelerators per node."""
    return dist.get_local_world_size() if cuda_available() else 0


def get_accel_model_name() -> str:
    """Query the accelerator name."""
    return accel_device_name(None) if cuda_available() else "N/A"


# Exception message and environment report
def _exc_report(exc_type) -> None:
    """Produces exception report (exception message + environment report)

    Args:
        exc_type (Exception): Type of exception.
    """

    # Don't print exception report for KeyboardInterrupt
    if not issubclass(exc_type, KeyboardInterrupt):
        print("\n-------------------\n" + EXCEPTION_MSG)
        print_env()


# Excepthook wrapper, wraps default excepthook and prints env info
def _custom_exception_handler(type, value, tb) -> None:
    """Custom exception wrapper for sys.excepthook."""
    sys.__excepthook__(type, value, tb)
    _exc_report(exc_type=type)


# Custom exception handler for IPython notebooks
def _nb_custom_exception_handler(self, type, value, tb, tb_offset=None):
    """Custom exception handler for IPython."""
    self.showtraceback((type, value, tb), tb_offset=tb_offset)  # standard IPython's printout
    _exc_report(exc_type=type)


# Public function to register excethook wrapper
def configure_excepthook() -> None:
    """Collect and print system information when :func:`sys.excepthook` is called.

    This function checks if the user is running from an IPython session and sets up the custom
    exception handler accordingly.

    To override the default :func:`sys.excepthook` with the custom except hook:

    .. testsetup::

        import sys
        from composer.utils.collect_env import configure_excepthook

    .. doctest::

        >>> configure_excepthook()
        >>> sys.excepthook 
        <function _custom_exception_handler at ...>
    """

    # Custom exceptions work differntly in notebooks
    if IPYTHON_AVAILABLE:
        # Set custom handler on Exception base class to apply to all exceptions
        nb.set_custom_exc((Exception,), _nb_custom_exception_handler)
    else:
        sys.excepthook = _custom_exception_handler


# Public function to restore original excepthook
def restore_excepthook() -> None:
    """Restore default :func:`sys.excepthook` behavior.

    Checks if user is running in an IPython notebook and restores the exception handler accordingly.

    To restore the default :func:`sys.__excepthook__`:

    .. testsetup::

        import sys
        from composer.utils.collect_env import restore_excepthook

    .. doctest::
        
        >>> restore_excepthook()
        >>> sys.excepthook
        <built-in function excepthook>
    """

    if IPYTHON_AVAILABLE:
        nb.set_custom_exc((Exception,), nb.showtraceback)
    else:
        sys.excepthook = sys.__excepthook__


# Get Torch environment info
def get_torch_env() -> str:
    """Query Torch system environment via :mod:`torch.utils.collect_env`."""
    return torchenv.get_pretty_env_info()


# Composer environment information string output format
composer_env_info_fmt = """
Composer version: {composer_version}
Number of nodes: {number_of_nodes}
Host processor model name: {host_processor_model_name}
Host processor core count: {host_processor_core_count}
Accelerators per node: {accelerators_per_node}
Accelerator model name: {accelerator_model_name}
""".strip()


# Get Composer environment info
def get_composer_env() -> str:
    """Query Composer pertinent system information."""

    mutable_dict = ComposerEnv(
        composer_version=get_composer_version(),
        number_of_nodes=get_num_nodes(),
        host_processor_model_name=get_host_processor_name(),
        host_processor_core_count=get_host_processor_cores(),
        accelerators_per_node=get_accel_per_node(),
        accelerator_model_name=get_accel_model_name(),
    )._asdict()

    return composer_env_info_fmt.format(**mutable_dict)


# Generate and print environment report
def print_env() -> None:
    """Generate system information report.

    Example:

    .. testsetup::

        from composer.utils import print_env

    .. doctest::

        >>> print_env()
        ---------------------------------
        System Environment Report
        ...
    """

    # Creation timestamp for report
    creation_time = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time()))
    report_hdr = f"---------------------------------\n" + \
                 f"System Environment Report        \n" + \
                 f"Created: {creation_time}\n"          + \
                 f"---------------------------------\n"

    # Torch section
    print(report_hdr)
    print("PyTorch information")
    print("-------------------")
    if TORCH_AVAILABLE:
        # Only print Torch system info if installed
        print(get_torch_env() + "\n\n")
    else:
        print("Torch not installed")

    print("Composer information")
    print("--------------------")
    if COMPOSER_AVAILABLE:
        # Only print Composer system info if installed
        print(get_composer_env() + "\n\n")
    else:
        print("Composer not installed")


# Invoke as standalone CLI script
def main() -> None:
    print("Collecting system information...")
    print_env()


if __name__ == '__main__':
    sys.exit(main())
