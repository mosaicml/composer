from argparse import ArgumentParser
import numpy as np
import os
from PIL import Image
import random
from torchvision.datasets import CIFAR10
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


def process_split(dataset, out_dir, n_shards, split):
    images, classes = get_shuffled(dataset)
    shard_size = len(images) // n_shards
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pattern = os.path.join(out_dir, f'{split}_%05d.tar')
    writer = ShardWriter(pattern, maxcount=shard_size)
    for sample_idx, (image, klass) in enumerate(zip(images, classes)):
        x = {
            '__key__': '%05d' % sample_idx,
            'jpg': image,
            'cls': klass,
        }
        writer.write(x)
    writer.close()


def main(args):
    dataset = CIFAR10(root="/datasets/cifar10", train=True, download=True)
    process_split(dataset, args.out_root, args.train_shards, 'train')

    dataset = CIFAR10(root="/datasets/cifar10", train=False, download=True)
    process_split(dataset, args.out_root, args.val_shards, 'val')


if __name__ == '__main__':
    main(parse_args())
