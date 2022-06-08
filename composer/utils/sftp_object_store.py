# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import sys
import tempfile
import uuid
import io
import warnings
from paramiko import SSHClient, SFTPAttributes
from typing import IO, Any, Dict, Iterator, Optional, Union


__all__ = ["SFTPObjectStore"]


class SFTPObjectStore:
    """Utility for uploading to and downloading to a server via SFTP
    """

    def __init__(self, host: str, port: int = 22, username: Optional[str] = None, key_file_path: Optional[str] = None):
        self.ssh_client = SSHClient()

        # add AutoAddPolicy
        if key_file_path is None:
            self.ssh_client.load_system_host_keys()
        else:
            try:
                self.ssh_client.load_system_host_keys(key_file_path)
            except IOError:
                warnings.warn("Host keys could not be read from {key_file_path}. Attempting to read keys from known hosts.")
                self.ssh_client.load_system_host_keys()
            
        # implement some retries here
        self.ssh_client.connect(host, port, username)

        self.sftp_client = self.ssh_client.open_sftp()

            
    def close_client(self):
        self.ssh_client.close()
        

    def upload_object(self,
                      file_path: str,
                      object_name: str):
        """Upload an object currently located on a disk.
        """
        # add verification and retries/rename
        tmp_path = object_name + f".{uuid.uuid4()}.tmp"
        
        object_size = os.stat(file_path).st_size
        attributes = (SFTPAttributes) (self.sftp_client.put(local_path=file_path, remotepath=tmp_path, confirm=True))
        uploaded_bytes = attributes.st_size

        if uploaded_bytes != object_size:
            self.sftp_client.remove(tmp_path)
            # Replace with custom exception later
            raise Exception("File sizes don't match uploading error")
        else:
            self.sftp_client.rename(tmp_path, object_name)


    def upload_object_via_stream(self,
                                 obj: Union[bytes, Iterator[bytes]],
                                 object_name: str,
                                 extra: Optional[Dict] = None,
                                 headers: Optional[Dict[str, str]] = None):
        """Upload an object.
        """
        obj = iter(obj)
        file_obj = 
        self.sftp_client.putfo(obj, )


    def download_object(self,
                        object_name: str,
                        destination_path: str):
        """Download an object to the specified destination path.


        Args:
            object_name (str): The name of the object to download.
            destination_path (str): Full path to a file or a directory where the incoming file will be saved.
        """
        # add verification and retries
        tmp_path = destination_path + f".{uuid.uuid4()}.tmp"

        object_size = self.sftp_client.stat(object_name).st_size
        
        self.sftp_client.get(remotepath=object_name, local_path=tmp_path)

        downloaded_bytes = os.stat(tmp_path).st_size

        if object_size != downloaded_bytes:
            os.remove(tmp_path)
            raise Exception
        else:
            os.replace(tmp_path, destination_path)



    def download_object_as_stream(self, object_name: str, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = os.path.join(tmpdir, )
        self.getfo()
