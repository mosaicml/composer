from argparse import ArgumentParser, Namespace
from glob import glob
from random import shuffle
from typing import Any, Dict, Generator, List, Tuple

from PIL import Image

from composer.datasets.webdataset import create_webdataset


def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--train_shards', type=int, default=512)
    args.add_argument('--val_shards', type=int, default=64)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def each_sample(pairs: List[Tuple[str, str]]) -> Generator[Dict[str, Any], None, None]:
    for idx, (scene_file, annotation_file) in enumerate(pairs):
        scene = Image.open(scene_file)
        annotation = Image.open(annotation_file)
        yield {
            '__key__': f'{idx:05d}',
            'scene.jpg': scene,
            'annotation.png': annotation,
        }


def process_split(in_root: str, out_root: str, split: str, n_shards: int, use_tqdm: int):
    pattern = f'{in_root}/images/{split}/ADE_{split}_*.jpg'
    scenes = sorted(glob(pattern))

    pattern = f'{in_root}/annotations/{split}/ADE_{split}_*.png'
    annotations = sorted(glob(pattern))

    pairs = list(zip(scenes, annotations))
    shuffle(pairs)

    create_webdataset(each_sample(pairs), out_root, split, len(pairs), n_shards, use_tqdm)


"""
Directory layout:

    ADE20k/
        annotations/
            train/
                ADE_train_%08d.png
            val/
                ADE_val_%08d.png
        images/
            test/
                ADE_test_%08d.jpg
            train/
                ADE_train_%08d.jpg
            val/
                ADE_val_%08d.jpg
"""


def main(args: Namespace) -> None:
    process_split(args.in_root, args.out_root, 'train', args.train_shards, args.tqdm)
    process_split(args.in_root, args.out_root, 'val', args.val_shards, args.tqdm)


if __name__ == '__main__':
    main(parse_args())
