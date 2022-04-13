import json
import logging
import os
import subprocess
import sys
import textwrap
from argparse import ArgumentParser

from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, ImageFolder
from composer.core.types import Dataset
from composer.datasets.ffcv_utils import write_ffcv_dataset

log = logging.getLogger(__name__)

def get_parser():
    parser = ArgumentParser(description="Utility for converting datasets to ffcv format.")

    parser.add_argument("--dataset",
                        type=str,
                        default="cifar",
                        choices=["cifar", "imagenet"],
                        help=textwrap.dedent("""\
                                Dataset to use. Default: cifar"""))
    parser.add_argument("--remote",
                        type=str,
                        default="s3://mosaicml-internal-dataset-cifar10",
                        help=textwrap.dedent("""\
                                WebDataset to use. Default: s3://mosaicml-internal-dataset-cifar10"""))
    parser.add_argument("--split",
                        type=str,
                        default="train",
                        choices=["train", "val"],
                        help=textwrap.dedent("""\
                                Split to use. Default: train"""))

    parser.add_argument("--datadir",
                        type=str,
                        default=None,
                        help=textwrap.dedent("""\
                                Location of the dataset. Default: None"""))

    parser.add_argument("--download",
                        type=bool,
                        default=False,
                        help=textwrap.dedent("""\
                                Download the dataset if possible. Default: False"""))

    parser.add_argument("--write_path",
                        type=str,
                        default=None,
                        help=textwrap.dedent("""\
                                File path to use for writing the dataset. Default: /tmp/<dataset>_<split>.ffcv"""))

    parser.add_argument("--write_mode",
                        type=str,
                        default="proportion",
                        choices=["raw", "jpg", "smart", "proportion"],
                        help=textwrap.dedent("""\
                                Write mode to use. raw is uint8 values, jpg is jpeg compressed images, smart is
                                compressing based on image size and proportion is according to the given
                                compress_probability. Default: proportion"""))

    parser.add_argument("--max_resolution", type=int, default=500, help="Max resoultion for images.")

    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers to use.")

    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size to use.")

    parser.add_argument("--jpeg_quality", type=int, default=90, help="Quality of jpeg.")

    parser.add_argument("--subset", type=int, default=-1, help="Only use a subset of dataset.")

    parser.add_argument("--compress_probability",
                        type=float,
                        required=False,
                        default=0.50,
                        help="Compress the given fraction of images to jpeg while writing the ffcv dataset.")
    return parser


def parse_args():
    parser = get_parser()

    args = parser.parse_args()

    if args.datadir is not None:
        log.info(f"Will read from local directory: {args.datadir}.")
    else:
        if args.remote.startswith('s3://'):
            log.info(f"Will read from remote: {args.remote}.")
        else:
            log.info(f"Will read from local: {args.remote}.")

    if args.write_path is None:
        args.write_path = f"/tmp/{args.dataset}_{args.split}.ffcv"

    return args


def main():

    args = parse_args()

    ds = None
    remote_location = None
    if args.datadir is not None:
        if args.dataset == 'cifar':
            ds = CIFAR10(root=args.datadir, train=(args.split == 'train'), download=args.download)
        elif args.dataset == 'imagenet':
            ds = ImageFolder(os.path.join(args.datadir, args.split))
        else:
            raise ValueError(f'Unsupported dataset: {args.dataset}. Checkout the list of supported datasets with -h')

        if args.subset > 0:
            ds = Subset(ds, range(args.subset))
    else:
        remote_location = os.path.join(args.remote, args.split)

    write_ffcv_dataset(dataset=ds,
                       remote=remote_location,
                       write_path=args.write_path,
                       max_resolution=args.max_resolution,
                       num_workers=args.num_workers,
                       write_mode=args.write_mode,
                       compress_probability=args.compress_probability,
                       jpeg_quality=args.jpeg_quality,
                       chunk_size=args.chunk_size)


if __name__ == '__main__':
    sys.exit(main())
