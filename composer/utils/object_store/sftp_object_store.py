# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import pathlib
import uuid
from typing import Callable, Optional, Union

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

        if '@' in host:
            self.username, self.host, *_ = self.host.split('@')
        try:
            from paramiko import SSHClient
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='paramiko') from e
        self.ssh_client = SSHClient()
        self.sftp_client = self._create_sftp_client()

    def _create_sftp_client(self):
        if self.key_file_path is None:
            #TODO change to read SSH config when key_file_path is None
            raise Exception("SFTPObjectStore initialized without a keyfile.")
        try:
            self.ssh_client.load_system_host_keys(self.key_file_path)
        except IOError:
            raise Exception('Host keys could not be read from {key_file_path}.')

        self.ssh_client.connect(self.host, self.port, self.username)
        if self.cwd is not None:
            self.ssh_client.exec_command(f'mkdir -p {self.cwd}')
            self.ssh_client.exec_command(f'cd {self.cwd}')

        self.sftp_client = self.ssh_client.open_sftp()

    def close(self):
        self.ssh_client.close()

    def get_uri(self, object_name: str) -> str:
        return f'sftp://{self.host}:{self.port}/{object_name}'

    def get_object_size(self, object_name: str) -> int:
        return self.sftp_client.stat(object_name).st_size

    def upload_object(self,
                      object_name: str,
                      filename: Union[str, pathlib.Path],
                      callback: Optional[Callable[[int, int], None]] = None) -> None:
        dirname = os.path.dirname(object_name)
        self.ssh_client.exec_command(f'mkdir -p {dirname}')
        try:
            self.sftp_client.put(filename, object_name, callback=callback, confirm=True)
        except IOError:
            raise ObjectStoreTransientError

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
        except IOError:
            # Make a best effort attempt to clean up the temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise ObjectStoreTransientError
        else:
            if overwrite:
                os.replace(tmp_path, filename)
            else:
                os.rename(tmp_path, filename)
