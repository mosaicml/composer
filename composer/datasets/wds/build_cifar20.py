from argparse import ArgumentParser
import numpy as np
import os
from PIL import Image
import random
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from webdataset import ShardWriter


def parse_args():
    x = ArgumentParser()
    x.add_argument('--out_root', type=str, required=True)
    x.add_argument('--train_shards', type=int, default=16)
    x.add_argument('--val_shards', type=int, default=8)
    return x.parse_args()


def get_shuffled(dataset):
    indices = np.random.permutation(len(dataset))
    images = dataset.data[indices]
    classes = np.array(dataset.targets)[indices]
    return images, classes


cifar100_to_cifar20 = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15,
    3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4,
    2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
])


def process_split(dataset, out_dir, n_shards, split):
    images, classes = get_shuffled(dataset)
    shard_size = len(images) // n_shards
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pattern = os.path.join(out_dir, f'{split}_%05d.tar')
    writer = ShardWriter(pattern, maxcount=shard_size)
    for sample_idx, (image, klass) in enumerate(zip(images, classes)):
        klass = cifar100_to_cifar20[klass]
        x = {
            '__key__': '%05d' % sample_idx,
            'jpg': image,
            'cls': klass,
        }
        writer.write(x)
    writer.close()


def main(args):
    dataset = CIFAR100(root="/datasets/cifar100", train=True, download=True)
    process_split(dataset, args.out_root, args.train_shards, 'train')

    dataset = CIFAR100(root="/datasets/cifar100", train=False, download=True)
    process_split(dataset, args.out_root, args.val_shards, 'val')


if __name__ == '__main__':
    main(parse_args())
