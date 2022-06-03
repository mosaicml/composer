# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""


from typing import Dict, Iterator, Optional, Union
import abc

__all__ = ["ObjectStore"]


class ObjectStore(abc.ABC):

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
        """
        pass

    def download_object_as_stream(self, object_name: str, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.
        """
        pass
