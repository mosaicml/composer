# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import pathlib
import urllib.parse
import uuid
from typing import Callable, Optional, Union

from paramiko import SSHException

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore, ObjectStoreTransientError

__all__ = ['SFTPObjectStore']


class SFTPObjectStore(ObjectStore):
    """Utility for uploading to and downloading to a server via SFTP.

    Args:
        host (str): The server to connect to. Also accepts a string in the form 'username@host'
        port (int, optional): The server port to connect to. Defaults to 22
        username (str, optional): The username to authenticate. Raises an exception if username is not
            passed in and also cannot be read from the host argument.
        key_file_path (str, optional): The filename of the private key.
        cwd (str, optional): The directory to navigate to upon creating the SSH connection. If not present
            it will be created.
    """

    def __init__(self,
                 host: str,
                 port: int = 22,
                 username: Optional[str] = None,
                 key_file_path: Optional[str] = None,
                 cwd: Optional[str] = None):
        self.host = host
        self.port = port
        self.username = username
        self.key_file_path = key_file_path
        self.cwd = cwd

        if host.startswith('sftp://'):
            url = urllib.parse.urlparse(host)
            if url.username:
                self.username = url.username
            if url.port:
                self.port = url.port
            if url.hostname:
                self.host = url.hostname
            if url.path != '/':
                dirname = os.path.dirname(url.path)
                if dirname != '/':
                    self.cwd = dirname
        try:
            from paramiko import SSHClient
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='paramiko') from e
        self.ssh_client = SSHClient()
        self.sftp_client = self._create_sftp_client()

    def _create_sftp_client(self):
        try:
            # Reads known host keys if self.key_file_path is NOne
            self.ssh_client.load_system_host_keys(self.key_file_path)
        except IOError:
            raise Exception('Host keys could not be read from {key_file_path}.')

        self.ssh_client.connect(self.host, self.port, self.username)
        sftp_client = self.ssh_client.open_sftp()
        if self.cwd is not None:
            sftp_client.mkdir(self.cwd)
            sftp_client.chdir(self.cwd)
        return sftp_client

    def close(self):
        self.ssh_client.close()

    def get_uri(self, object_name: str) -> str:
        base = f'sftp://{self.host}:{self.port}/'
        if self.cwd is not None:
            return os.path.join(base, self.cwd, object_name)
        return os.path.join(base, object_name)

    def get_object_size(self, object_name: str) -> int:
        st_size = self.sftp_client.stat(object_name).st_size
        if st_size is None:
            raise RuntimeError
        return st_size

    def upload_object(self,
                      object_name: str,
                      filename: Union[str, pathlib.Path],
                      callback: Optional[Callable[[int, int], None]] = None) -> None:
        dirname = os.path.dirname(object_name)
        self.sftp_client.mkdir(dirname)
        try:
            self.sftp_client.put(str(filename), object_name, callback=callback, confirm=True)
        except SSHException:
            raise ObjectStoreTransientError
        except:
            raise

    def download_object(self,
                        object_name: str,
                        filename: Union[str, pathlib.Path],
                        overwrite: bool = False,
                        callback: Optional[Callable[[int, int], None]] = None) -> None:
        dirname = os.path.dirname(filename)
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
            if isinstance(e, SSHException):
                raise ObjectStoreTransientError
            else:
                raise
        else:
            if overwrite:
                os.replace(tmp_path, filename)
            else:
                os.rename(tmp_path, filename)
