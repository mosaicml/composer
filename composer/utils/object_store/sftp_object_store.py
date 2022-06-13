# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import uuid
from typing import Optional

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStoreTransientError

__all__ = ["SFTPObjectStore"]


class SFTPObjectStore:
    """Utility for uploading to and downloading to a server via SFTP.
    
    Args:
        host (str): The server to connect to
        port (int): The server port to connect to
        username (str): The username to authenticate
        key_file_path (str): The filename of private key
        cwd (Optional[str]): The directory to navigate to upon creating the SSH connection.
    """

    #TODO - read creds automatically through envvariables

    def __init__(self,
                 host: str,
                 port: int,
                 username: str,
                 key_file_path: str,
                 cwd: Optional[str] = None):
        self.host = host
        self.port = port
        self.username = username
        self.key_file_path = key_file_path
        self.cwd = cwd

        try:
            from paramiko import SSHClient
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group="streaming", conda_package="paramiko") from e
        self.ssh_client = SSHClient()

        try:
            self.ssh_client.load_system_host_keys(key_file_path)
        except IOError:
            raise Exception("Host keys could not be read from {key_file_path}.")

        self.ssh_client.connect(host, port, username)
        if cwd is not None:
            self.ssh_client.exec_command(f"mkdir -p {cwd}")
            self.ssh_client.exec_command(f"cd {cwd}")

        self.sftp_client = self.ssh_client.open_sftp()

    def close(self):
        self.ssh_client.close()
    
    def get_uri(self, object_name: str) -> str:
        return f'sftp://{self.host}:{self.port}/{object_name}'

    def upload_object(self, file_path: str, object_name: str):
        tmp_path = object_name + f".{uuid.uuid4()}.tmp"

        object_size = os.stat(file_path).st_size
        attributes = self.sftp_client.put(file_path, tmp_path, confirm=True)
        uploaded_bytes = attributes.st_size

        if uploaded_bytes != object_size:
            self.sftp_client.remove(tmp_path)
            raise ObjectStoreTransientError
        else:
            self.sftp_client.rename(tmp_path, object_name)

    def download_object(self, object_name: str, destination_path: str):
        dirname = os.path.dirname(destination_path)
        os.makedirs(dirname, exist_ok=True)

        tmp_path = destination_path + f".{uuid.uuid4()}.tmp"

        object_size = self.sftp_client.stat(object_name).st_size

        self.sftp_client.get(remotepath=object_name, localpath=tmp_path)

        downloaded_bytes = os.stat(tmp_path).st_size

        if object_size != downloaded_bytes:
            os.remove(tmp_path)
            raise ObjectStoreTransientError
        else:
            os.replace(tmp_path, destination_path)
