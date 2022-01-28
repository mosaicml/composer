from argparse import ArgumentParser
from glob import glob
import os
from PIL import Image
from random import shuffle
from tqdm import tqdm
from webdataset import ShardWriter


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in_root', type=str, required=True)
    x.add_argument('--out_root', type=str, required=True)
    x.add_argument('--train_shards', type=int, default=1024)
    x.add_argument('--val_shards', type=int, default=128)
    return x.parse_args()


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


def save(pairs, out_dir, split, shard_size):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pattern = os.path.join(out_dir, f'{split}_%05d.tar')
    writer = ShardWriter(pattern, maxcount=shard_size)
    for sample_idx, (filename, klass) in enumerate(tqdm(pairs, leave=False)):
        image = Image.open(filename).convert('RGB')
        x = {
            '__key__': '%05d' % sample_idx,
            'jpg': image,
            'cls': klass,
        }
        writer.write(x)
    writer.close()


def main(args):
    pairs = find(args.in_root, 'train')
    shard_size = len(pairs) // args.train_shards
    save(pairs, args.out_root, 'train', shard_size)

    pairs = find(args.in_root, 'val')
    shard_size = len(pairs) // args.val_shards
    save(pairs, args.out_root, 'val', shard_size)


if __name__ == '__main__':
    main(parse_args())
