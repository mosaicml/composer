import json
import logging
import subprocess
import textwrap
from typing import Optional

from composer.core.types import Dataset

log = logging.getLogger(__name__)

__all__ = ["write_ffcv_dataset"]


def write_ffcv_dataset(dataset: Optional[Dataset] = None,
                       remote: Optional[str] = None,
                       write_path: str = "/tmp/dataset.ffcv",
                       max_resolution: Optional[int] = None,
                       num_workers: int = 16,
                       write_mode: str = 'raw',
                       compress_probability: float = 0.50,
                       jpeg_quality: float = 90,
                       chunk_size: int = 100):
    """Converts PyTorch ``dataset`` or webdataset at ``remote`` into FFCV format at filepath ``write_path``.

    Args:
        dataset (Iterable[Sample]): A PyTorch dataset. Default: ``None``.
        remote (str): A remote path for webdataset. Default: ``None``.
        write_path (str): Write results to this file. Default: ``"/tmp/dataset.ffcv"``.
        max_resolution (int): Limit resolution if provided. Default: ``None``.
        num_workers (int): Numbers of workers to use. Default: ``16``.
        write_mode (str): Write mode for the dataset. Default: ``'raw'``.
        compress_probability (float): Probability with which image is JPEG-compressed. Default: ``0.5``.
        jpeg_quality (float): Quality to use for jpeg compression. Default: ``90``.
        chunk_size (int): Size of chunks processed by each worker during conversion. Default: ``100``.
    """
    try:
        import ffcv  # type: ignore
    except ImportError:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without ffcv support.
            To use ffcv with Composer, please install ffcv in your environment."""))

    if dataset is None and remote is None:
        raise ValueError("At least one of dataset or remote should not be None.")

    log.info(f"Writing dataset in FFCV <file>.ffcv format to {write_path}.")
    writer = ffcv.writer.DatasetWriter(write_path, {
        'image':
            ffcv.fields.RGBImageField(write_mode=write_mode,
                                      max_resolution=max_resolution,
                                      compress_probability=compress_probability,
                                      jpeg_quality=jpeg_quality),
        'label':
            ffcv.fields.IntField()
    },
                                       num_workers=num_workers)
    if dataset:
        writer.from_indexed_dataset(dataset, chunksize=chunk_size)
    else:
        pipeline = lambda dataset: dataset.decode('pil').to_tuple('jpg', 'cls')

        metadata_url = f'{remote}/meta.json'
        cmd = 'aws', 's3', 'cp', metadata_url, '-'
        ret = subprocess.run(cmd, capture_output=True)
        assert not ret.stderr, 'Download failed, check your credentials?'
        text = ret.stdout

        metadata = json.loads(text)
        num_shards = metadata['n_shards']
        if metadata['n_leftover'] > 0:
            num_shards = num_shards + 1
        urls = [f'pipe: aws s3 cp {remote}/{idx:05d}.tar -' for idx in range(num_shards)]

        writer.from_webdataset(urls, pipeline)
