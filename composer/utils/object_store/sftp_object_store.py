# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""

from __future__ import annotations

import contextlib
import os
import pathlib
import urllib.parse
import uuid
from typing import Any, Callable, Optional, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore, ObjectStoreTransientError

__all__ = ['SFTPObjectStore']

try:
    import paramiko.client
    from paramiko import SSHClient
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False


def _set_kwarg(value: Any, kwargs: dict[str, Any], arg_name: str, kwarg_name: str):
    if kwarg_name in kwargs:
        raise ValueError(f'The `{arg_name}` should be not be specified directly if also included via `connect_kwargs`')
    kwargs[kwarg_name] = value


class SFTPObjectStore(ObjectStore):
    """Utility for uploading to and downloading to a server via SFTP.

    Args:
        host (str): The server to connect to.

            Also accepts a URI string in the form ``'sftp://username@host:port/./relative/path'``.
            For an absolute path, use a double `//` -- e.g. ``'sftp://username@host:port//absolute/path'``.

        port (int, optional): The server port to connect to.
        username (str, optional): The username (if not specified in the SSH config) needed to authenticate.
            Defaults to None.
        password (str, optional): The password (if required) needed to authenticate. Defaults to None.
        key_filename (pathlib.Path | str, optional): The filepath to the a private key (if required) needed to
            authenticate. Defaults to None. Any keys specified here will be tried *in addition* to any keys
            specified in ``~/.ssh/`` or via a SSH agent.
        key_filename_environ (str, optional): Environment variable defining path to an SSH keyfile.
            Only used if key_filename is not provided. Default: ``COMPOSER_SFTP_KEY_FILE``.
        known_hosts_filename (pathlib.Path | str, optional): The filename of the known hosts file. If not specified,
            the default SSH known hosts will be used.
        known_hosts_filename_environ (str, optional). Environment variable defining path to a known hosts file.
            Only used if known_hosts_filename is not provided. Default: ``COMPOSER_SFTP_KNOWN_HOSTS_FILE``.
        missing_host_key_policy (str | paramiko.client.MissingHostKeyPolicy, optional): The class name or instance of
            :class:`paramiko.client.MissingHostKeyPolicy` to use for a missing host key. Defaults to ``'RejectPolicy'``.

            Built-in options:
            *   ``'RejectPolicy'`` (the default), which will reject any host key not authorized in the ``known_hosts_filename``.
            *   ``'AutoAddPolicy'``, which will add any unknown host key.
            *   ``'WarningPolicy'``, which will warn on an unknown host key.

            For custom logic, subclass :class:`paramiko.client.MissingHostKeyPolicy`, and provide an instance of this class.
        cwd (str, optional): The directory to navigate to upon creating the SSH connection. If not present
            it will be created.
        connect_kwargs (dict[str, Any], optional): Any additional kwargs to pass through to :meth:`.SSHClient.connect`.
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: Optional[str] = None,
        password: Optional[str] = None,
        known_hosts_filename: Optional[Union[pathlib.Path, str]] = None,
        known_hosts_filename_environ: str = 'COMPOSER_SFTP_KNOWN_HOSTS_FILE',
        key_filename: Optional[Union[pathlib.Path, str]] = None,
        key_filename_environ: str = 'COMPOSER_SFTP_KEY_FILE',
        missing_host_key_policy: Union[str, paramiko.client.MissingHostKeyPolicy] = 'RejectPolicy',
        cwd: str = '',
        connect_kwargs: Optional[dict[str, Any]] = None,
    ):

        if known_hosts_filename is None:
            known_hosts_filename = os.environ.get(known_hosts_filename_environ, None)
        self.known_hosts_filename = known_hosts_filename

        if key_filename is None:
            key_filename = os.environ.get(key_filename_environ, None)
        self.key_filename = key_filename

        if not _PARAMIKO_AVAILABLE:
            raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='paramiko')
        url = urllib.parse.urlsplit(host)
        if url.scheme != '':
            if url.scheme.lower() != 'sftp':
                raise ValueError('If specifying a URI, only the sftp scheme is supported.')
            if not url.hostname:
                raise ValueError('If specifying a URI, the URI must include the hostname.')
            host = url.hostname
            if url.username:
                if username is not None:
                    raise ValueError(
                        'If specifying the username in the `host`, then the `username` argument must be blank.',
                    )
                username = url.username
            if url.password:
                if password is not None:
                    raise ValueError(
                        'If specifying the password in the `host`, then the `password` argument must be blank.',
                    )
                password = url.password
            if url.port:
                if port != 22:
                    raise ValueError('If specifying the port in the `host`, then the `port` argument must be blank.')
                port = url.port
            if url.path:
                # strip the first left slash. Two slashes for absolute; 1 for relative
                assert url.path.startswith('/'), 'The path should always start with a `/`'
                cwd = url.path[1:]
            if url.query or url.fragment:
                raise ValueError('Query and fragment parameters are not supported as part of a URI.')
        if connect_kwargs is None:
            connect_kwargs = {}
        if host:
            _set_kwarg(host, connect_kwargs, arg_name='host', kwarg_name='hostname')
        if port:
            _set_kwarg(port, connect_kwargs, arg_name='port', kwarg_name='port')
        if username:
            _set_kwarg(username, connect_kwargs, arg_name='username', kwarg_name='username')
        if password:
            _set_kwarg(password, connect_kwargs, arg_name='password', kwarg_name='password')
        if key_filename:
            _set_kwarg(key_filename, connect_kwargs, arg_name='key_filename', kwarg_name='key_filename')

        if cwd and not cwd.endswith('/'):
            cwd += '/'
        self.cwd = cwd

        netloc = ''
        if username:
            netloc += f'{username}@'
        if host:
            netloc += host
        if port:
            netloc += f':{port}'

        self._base_uri = urllib.parse.urlunsplit((
            'sftp',  # scheme
            netloc,  # netloc
            '/' + cwd,  # path
            None,  # query
            None,  # fragment
        ))
        self.ssh_client = SSHClient()
        if known_hosts_filename is not None:
            known_hosts_filename = str(known_hosts_filename)
        if isinstance(missing_host_key_policy, str):
            try:
                missing_host_key_policy = getattr(paramiko.client, missing_host_key_policy)()
                assert isinstance(missing_host_key_policy, paramiko.client.MissingHostKeyPolicy)
            except AttributeError:
                raise ValueError(
                    "Invalid `missing_host_key_policy`. Must be 'AutoAddPolicy', 'RejectPolicy', or 'WarningPolicy'.",
                )
        self.ssh_client.set_missing_host_key_policy(missing_host_key_policy)
        self.ssh_client.load_system_host_keys(known_hosts_filename)
        self._connect_kwargs = connect_kwargs
        self.ssh_client.connect(**connect_kwargs)
        self.sftp_client = self.ssh_client.open_sftp()

    def close(self):
        self.sftp_client.close()
        self.ssh_client.close()

    def get_uri(self, object_name: str) -> str:
        return self._base_uri + object_name

    def get_object_size(self, object_name: str) -> int:
        object_name = os.path.join(self.cwd, object_name)
        with self._handle_transient_errors():
            st_size = self.sftp_client.stat(object_name).st_size
        if st_size is None:
            raise RuntimeError('Cannot determine object size: stat(object_name).st_size is None')
        return st_size

    @contextlib.contextmanager
    def _handle_transient_errors(self):
        from paramiko import ChannelException, SSHException
        try:
            yield
        except Exception as e:
            if not self._is_cnx_alive():
                # If the connection dropped, then it's a transient error. Create a new one, and raise the exception to try again.
                self.close()
                self.ssh_client.connect(**self._connect_kwargs)
                self.sftp_client = self.ssh_client.open_sftp()
                raise ObjectStoreTransientError from e
            if isinstance(e, SSHException):
                if 'Server connection dropped:' in str(e):
                    raise ObjectStoreTransientError from e
            if isinstance(e, (TimeoutError, ConnectionError, EOFError, ChannelException)):
                raise ObjectStoreTransientError from e
            raise e

    def _is_cnx_alive(self):
        transport = self.ssh_client.get_transport()
        assert transport is not None, 'transport should not be None'
        if not transport.is_active() or not transport.is_alive():
            return False
        channel = self.sftp_client.get_channel()
        assert channel is not None, 'channels not be None if the transport is alive'
        return channel.active and not channel.closed

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        remote_object_name = os.path.join(self.cwd, object_name)
        dirname = os.path.dirname(remote_object_name)
        with self._handle_transient_errors():
            if dirname:
                self.ssh_client.exec_command(f'mkdir -p {dirname}')
            self.sftp_client.put(str(filename), remote_object_name, callback=callback, confirm=False)
            # Validating manually to raise ObjectStoreTransientErrors if the size mismatches
            # This logic was adapted from the original source -- see
            # https://github.com/paramiko/paramiko/blob/1824a27c644132e5d46f2294c1e2fa131c523559/paramiko/sftp_client.py#L719-L724
            local_file_size = os.stat(filename).st_size
            remote_file_size = self.get_object_size(object_name)
            if local_file_size != remote_file_size:
                raise ObjectStoreTransientError(
                    f'Size mismatch in put: local size ({local_file_size}) != remote size ({remote_file_size})',
                )

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        object_name = os.path.join(self.cwd, object_name)
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists')

        tmp_path = str(filename) + f'.{uuid.uuid4()}.tmp'

        try:
            with self._handle_transient_errors():
                self.sftp_client.get(remotepath=object_name, localpath=tmp_path, callback=callback)
        except Exception:
            # Make a best effort attempt to clean up the temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        else:
            if overwrite:
                os.replace(tmp_path, filename)
            else:
                os.rename(tmp_path, filename)
