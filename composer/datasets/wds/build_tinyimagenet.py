from argparse import ArgumentParser
import os
from PIL import Image
import random
from tqdm import tqdm
from webdataset import ShardWriter


'''
Directory layout:

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

    web_tinyimagenet/
        train_{shard}.tar
        val_{shard}.tar
'''


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in_root', type=str, required=True)
    x.add_argument('--out_root', type=str, required=True)
    x.add_argument('--train_shards', type=int, default=16)
    x.add_argument('--val_shards', type=int, default=8)
    return x.parse_args()


def get_train(in_root, wnids):
    pairs = []
    for wnid_idx, wnid in tqdm(enumerate(wnids), leave=False):
        in_dir = os.path.join(in_root, 'train', wnid, 'images')
        for basename in os.listdir(in_dir):
            filename = os.path.join(in_dir, basename)
            pairs.append((filename, wnid_idx))
    random.shuffle(pairs)
    return pairs


def get_val(in_root, wnid2idx):
    pairs = []
    filename = os.path.join(in_root, 'val',  'val_annotations.txt')
    lines = open(filename).read().strip().split('\n')
    for line in tqdm(lines, leave=False):
        basename, wnid = line.split()[:2]
        filename = os.path.join(in_root, 'val', 'images', basename)
        wnid_idx = wnid2idx[wnid]
        pairs.append((filename, wnid_idx))
    random.shuffle(pairs)
    return pairs


def save(pairs, out_dir, split, shard_size):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pattern = os.path.join(out_dir, f'{split}_%05d.tar')
    writer = ShardWriter(pattern, maxcount=shard_size)
    for sample_idx, (filename, wnid_idx) in enumerate(pairs):
        image = Image.open(filename)
        x = {
            '__key__': '%05d' % sample_idx,
            'jpg': image,
            'cls': wnid_idx,
        }
        writer.write(x)
    writer.close()


def main(args):
    filename = os.path.join(args.in_root, 'wnids.txt')
    wnids = open(filename).read().split()
    wnid2idx = dict(zip(wnids, range(len(wnids))))

    pairs = get_train(args.in_root, wnids)
    shard_size = len(pairs) // args.train_shards
    save(pairs, args.out_root, 'train', shard_size)

    pairs = get_val(args.in_root, wnid2idx)
    shard_size = len(pairs) // args.val_shards
    save(pairs, args.out_root, 'val', shard_size)


if __name__ == '__main__':
    main(parse_args())
