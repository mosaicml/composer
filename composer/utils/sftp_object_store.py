# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import sys
import tempfile
import uuid
from paramiko import RSAKey, SFTPClient, Transport
from typing import Any, Dict, Iterator, Optional, Union


__all__ = ["SFTPObjectStore"]


class SFTPObjectStore:
    """Utility for uploading to and downloading to a server via SFTP
    """

    def __init__(self, host: str, port: int, username: str, key_file_path: str):
        try:
            key = RSAKey.from_private_key_file(key_file_path)
            transport = Transport(sock=(host, port))
            #host key verification?
            transport.connect(None, username=username, pkey=key)

            self.client = SFTPClient.from_transport(transport)
            

        except Exception as e:
            if self.client is not None:
                self.client.close()
            if transport is not None:
                transport.close()
            
    def close(self):
        if self.client is not None:
                self.client.close()
            if transport is not None:
                transport.close()

    def __init__(self, provider: str, container: str, provider_kwargs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def _create_client(self, host, port, username, password, keyfilepath, keyfiletype):
        pass
        

    def upload_object(self,
                      file_path: str,
                      object_name: str,
                      verify_hash: bool = True,
                      extra: Optional[Dict] = None,
                      headers: Optional[Dict[str, str]] = None):
        """Upload an object currently located on a disk.
        """
        pass

    def upload_object_via_stream(self,
                                 obj: Union[bytes, Iterator[bytes]],
                                 object_name: str,
                                 extra: Optional[Dict] = None,
                                 headers: Optional[Dict[str, str]] = None):
        """Upload an object.
        """
        pass

    def download_object(self,
                        object_name: str,
                        destination_path: str,
                        overwrite_existing: bool = False,
                        delete_on_failure: bool = True):
        """Download an object to the specified destination path.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.download_object`.

        Args:
            object_name (str): The name of the object to download.

            destination_path (str): Full path to a file or a directory where the incoming file will be saved.

            overwrite_existing (bool, optional): Set to ``True`` to overwrite an existing file. (default: ``False``)
            delete_on_failure (bool, optional): Set to ``True`` to delete a partially downloaded file if
                the download was not successful (hash mismatch / file size). (default: ``True``)
        """
        pass

    def download_object_as_stream(self, object_name: str, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.
        """
        pass
        # 
