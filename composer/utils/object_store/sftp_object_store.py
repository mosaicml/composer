# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import pathlib
import urllib.parse
import uuid
from typing import Any, Callable, Dict, Optional, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore, ObjectStoreTransientError

__all__ = ['SFTPObjectStore']

try:
    from paramiko import SSHClient
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False


def _set_kwarg(value: Any, kwargs: Dict[str, Any], arg_name: str, kwarg_name: str):
    if kwarg_name in kwargs:
        raise ValueError(f'The `{arg_name}` should be not be specified directly if also included via `connect_kwargs`')
    kwargs[kwarg_name] = value


def _ensure_transient_errors_are_wrapped(e: Exception):
    from paramiko import SSHException
    if isinstance(e, SSHException):
        if 'Server connection dropped:' in str(e):
            raise ObjectStoreTransientError from e
    raise e


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
        known_hosts_filename (pathlib.Path | str, optional): The filename of the known hosts file. If not specified,
            the default SSH known hosts will be used.
        cwd (str, optional): The directory to navigate to upon creating the SSH connection. If not present
            it will be created.
        connect_kwargs (Dict[str, Any], optional): Any additional kwargs to pass through to :meth:`.SSHClient.connect`.
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: Optional[str] = None,
        password: Optional[str] = None,
        known_hosts_filename: Optional[Union[pathlib.Path, str]] = None,
        key_filename: Optional[Union[pathlib.Path, str]] = None,
        cwd: Optional[str] = None,
        connect_kwargs: Optional[Dict[str, Any]] = None,
    ):
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
                        'If specifying the username in the `host`, then the `username` argument must be blank.')
                username = url.username
            if url.password:
                if password is not None:
                    raise ValueError(
                        'If specifying the password in the `host`, then the `password` argument must be blank.')
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

        if cwd is not None:
            if not cwd.endswith('/'):
                cwd += '/'

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
            '/' + (cwd or ''),  # path
            None,  # query
            None,  # fragment
        ))
        self.ssh_client = SSHClient()
        if known_hosts_filename is not None:
            known_hosts_filename = str(known_hosts_filename)
        self.ssh_client.load_system_host_keys(known_hosts_filename)
        self.ssh_client.connect(**connect_kwargs)
        self.sftp_client = self.ssh_client.open_sftp()
        if cwd:
            self.ssh_client.exec_command(f'mkdir -p {cwd}')
            self.sftp_client.chdir(cwd)

    def close(self):
        self.sftp_client.close()
        self.ssh_client.close()

    def get_uri(self, object_name: str) -> str:
        return self._base_uri + object_name

    def get_object_size(self, object_name: str) -> int:
        st_size = self.sftp_client.stat(object_name).st_size
        if st_size is None:
            raise RuntimeError('Cannot determine object size: stat(object_name).st_size is None')
        return st_size

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        dirname = os.path.dirname(object_name)
        if dirname:
            self.ssh_client.exec_command(f'mkdir -p {dirname}')
        try:
            self.sftp_client.put(str(filename), object_name, callback=callback, confirm=True)
        except Exception as e:
            _ensure_transient_errors_are_wrapped(e)

    def download_object(self,
                        object_name: str,
                        filename: Union[str, pathlib.Path],
                        overwrite: bool = False,
                        callback: Optional[Callable[[int, int], None]] = None) -> None:
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists')

        tmp_path = str(filename) + f'.{uuid.uuid4()}.tmp'

        try:
            self.sftp_client.get(remotepath=object_name, localpath=tmp_path, callback=callback)
        except Exception as e:
            # Make a best effort attempt to clean up the temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            _ensure_transient_errors_are_wrapped(e)
        else:
            if overwrite:
                os.replace(tmp_path, filename)
            else:
                os.rename(tmp_path, filename)
