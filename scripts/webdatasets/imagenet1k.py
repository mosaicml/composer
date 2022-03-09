import os
from argparse import ArgumentParser, Namespace
from glob import glob
from random import shuffle
from typing import Any, Dict, Generator, List, Tuple

from composer.datasets.webdataset import create_webdataset


def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--train_shards', type=int, default=1024)
    args.add_argument('--val_shards', type=int, default=128)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def find_samples(in_root: str, split: str) -> List[Tuple[str, int]]:
    pattern = os.path.join(in_root, split, '*', '*.JPEG')
    filenames = sorted(glob(pattern))
    wnid2class = {}
    pairs = []
    for filename in filenames:
        parts = filename.split(os.path.sep)
        wnid = parts[-2]
        cls = wnid2class.get(wnid)
        if cls is None:
            cls = len(wnid2class)
            wnid2class[wnid] = cls
        pairs.append((filename, cls))
    shuffle(pairs)
    return pairs


def each_sample(pairs: List[Tuple[str, int]]) -> Generator[Dict[str, Any], None, None]:
    for idx, (img_file, cls) in enumerate(pairs):
        img = open(img_file, 'rb').read()
        yield {
            '__key__': f'{idx:05d}',
            'jpg': img,
            'cls': cls,
        }


def main(args: Namespace) -> None:
    pairs = find_samples(args.in_root, 'train')
    create_webdataset(each_sample(pairs), args.out_root, 'train', len(pairs), args.train_shards, args.tqdm)

    pairs = find_samples(args.in_root, 'val')
    create_webdataset(each_sample(pairs), args.out_root, 'val', len(pairs), args.val_shards, args.tqdm)


if __name__ == '__main__':
    main(parse_args())
