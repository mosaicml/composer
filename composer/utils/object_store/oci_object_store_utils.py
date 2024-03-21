# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for monkey patching OCI."""

import logging
import threading

from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)


def patch_oci():
    """Monkey patches the OCI retry strategy to add BaseRequestException.

    This is necessary because the OCI retry strategy does not handle BaseRequestException.
    """
    try:
        import oci
    except ImportError as e:
        raise MissingConditionalImportError(
            conda_package='oci',
            extra_deps_group='oci',
            conda_channel='conda-forge',
        ) from e
    oci.retry.retry_checkers.TimeoutConnectionAndServiceErrorRetryChecker.should_retry = should_retry


def should_retry(self, exception=None, response=None, **kwargs):
    """Uses OCI retry and adds BaseRequestException.

    https://github.com/oracle/oci-python-sdk/blob/8ba39d63242fd406fb7a97fb6f9acdde81eb6dd9/src/oci/retry/retry_checkers.py#L166
    """
    log.warning('Using monkeypatched should_retry')
    try:
        from circuitbreaker import CircuitBreakerError
        from oci._vendor.requests.exceptions import ConnectionError as RequestsConnectionError
        from oci._vendor.requests.exceptions import RequestException as BaseRequestException
        from oci._vendor.requests.exceptions import Timeout
        from oci.exceptions import ConnectTimeout, RequestException, ServiceError
    except ImportError as e:
        raise MissingConditionalImportError(
            conda_package='oci',
            extra_deps_group='oci',
            conda_channel='conda-forge',
        ) from e
    print(type(exception))
    print(str(exception))
    if isinstance(exception, Timeout):
        return True
    elif isinstance(exception, RequestsConnectionError):
        return True
    elif isinstance(exception, RequestException):
        return True
    elif isinstance(exception, BaseRequestException):
        return True
    elif isinstance(exception, ConnectTimeout):
        return True
    elif isinstance(exception, CircuitBreakerError):
        if 'circuit_breaker_callback' in kwargs:
            threading.Thread(target=kwargs['circuit_breaker_callback'], args=(exception,)).start()
        return True
    elif isinstance(exception, ServiceError):
        if exception.status in self.service_error_retry_config:  # type: ignore
            codes = self.service_error_retry_config[exception.status]  # type: ignore
            if not codes:
                return True
            else:
                return exception.code in codes  # type: ignore
        elif self.retry_any_5xx and exception.status >= 500 and exception.status != 501:  # type: ignore
            return True
    else:
        # This is inside a try block because ConnectionError exists in Python 3 and not in Python 2
        try:
            if isinstance(exception, ConnectionError):
                return True
        except NameError:
            pass

    return False
