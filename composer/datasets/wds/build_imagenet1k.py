from argparse import ArgumentParser
from glob import glob
import os
from PIL import Image
from random import shuffle

from composer.datasets.webdataset import create_webdataset


def parse_args():
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--train_shards', type=int, default=1024)
    args.add_argument('--val_shards', type=int, default=128)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def find(in_root, split):
    pattern = os.path.join(in_root, split, '*', '*.JPEG')
    filenames = sorted(glob(pattern))
    wnid2class = {}
    pairs = []
    for filename in filenames:
        parts = filename.split(os.path.sep)
        wnid = parts[-2]
        klass = wnid2class.get(wnid)
        if klass is None:
            klass = len(wnid2class)
            wnid2class[wnid] = wnid
        pairs.append((filename, klass))
    shuffle(pairs)
    return pairs


def each_sample(pairs):
    for idx, (img_file, cls) in enumerate(pairs):
        img = Image.open(img_file).convert('RGB')
        yield {
            '__key__': f'{idx:05d}',
            'image': img,
            'class': cls,
        }


def main(args):
    pairs = find(args.in_root, 'train')
    create_webdataset(each_sample(pairs), args.out_root, 'train', len(pairs), args.train_shards,
                      args.tqdm)

    pairs = find(args.in_root, 'val')
    create_webdataset(each_sample(pairs), args.out_root, 'val', len(pairs), args.val_shards,
                      args.tqdm)


if __name__ == '__main__':
    main(parse_args())
