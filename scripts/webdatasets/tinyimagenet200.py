import os
from argparse import ArgumentParser, Namespace
from random import shuffle
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image
from tqdm import tqdm

from composer.datasets.webdataset import create_webdataset
"""
Input directory layout:

    tiny-imagenet-200/
        test/
            images/
                (10k images)
        train/
            (200 wnids)/
                (500 images per dir)
        val/
            images/
                (10k images)
            val_annotations.txt  # 10k rows of (file, wnid, x, y, h, w)
        wnids.txt  # 200 rows of (wnid)
        words.txt  # 82115 rows of (wnid, wordnet category name)
"""


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--train_shards', type=int, default=128)
    args.add_argument('--val_shards', type=int, default=128)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get_train(in_root: str, wnids: List[str]) -> List[Tuple[str, int]]:
    """Get training split sample pairs.

    Args:
        in_root (str): Input dataset root directory.
        wnids (list): Ordered list of WordNet IDs.

    Returns:
        List of pairs of (image filename, class ID).
    """
    pairs = []
    for wnid_idx, wnid in tqdm(enumerate(wnids), leave=False):
        in_dir = os.path.join(in_root, 'train', wnid, 'images')
        for basename in os.listdir(in_dir):
            filename = os.path.join(in_dir, basename)
            pairs.append((filename, wnid_idx))
    shuffle(pairs)
    return pairs


def get_val(in_root: str, wnid2idx: Dict[str, int]) -> List[Tuple[str, int]]:
    """Get validation split sample pairs.

    Args:
        in_root (str): Input dataset root directory.
        wnid2idx (dict): Mapping of WordNet ID to class ID.

    Returns:
        List of pairs of (image filename, class ID).
    """
    pairs = []
    filename = os.path.join(in_root, 'val', 'val_annotations.txt')
    lines = open(filename).read().strip().split('\n')
    for line in tqdm(lines, leave=False):
        basename, wnid = line.split()[:2]
        filename = os.path.join(in_root, 'val', 'images', basename)
        wnid_idx = wnid2idx[wnid]
        pairs.append((filename, wnid_idx))
    shuffle(pairs)
    return pairs


def each_sample(pairs: List[Tuple[str, int]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        pairs (list of (str, int)): List of pairs of (image filename, class ID).

    Yields:
        Sample dicts.
    """
    for idx, (img_file, cls) in enumerate(pairs):
        img = Image.open(img_file)
        yield {
            '__key__': f'{idx:05d}',
            'jpg': img,
            'cls': cls,
        }


def main(args: Namespace) -> None:
    """Main: create tinyimagenet200 webdataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    filename = os.path.join(args.in_root, 'wnids.txt')
    wnids = open(filename).read().split()
    wnid2idx = dict(zip(wnids, range(len(wnids))))

    pairs = get_train(args.in_root, wnids)
    create_webdataset(each_sample(pairs), args.out_root, 'train', len(pairs), args.train_shards, args.tqdm)

    pairs = get_val(args.in_root, wnid2idx)
    create_webdataset(each_sample(pairs), args.out_root, 'val', len(pairs), args.val_shards, args.tqdm)


if __name__ == '__main__':
    main(parse_args())
