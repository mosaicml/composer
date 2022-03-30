"""
Script to measure empirically how fairly the samples of webdatasets with imbalanced shards are represented in training.

This is done by applying a specially crafted model to a specially crafted dataset. The model is just a matrix of
weights initialized to 1/n. The samples must each be different and each provide zero information on the others. We do
this by having the samples be the one-hot sample ID mapped to itself.

As the model is trained, the weight matrix is expected to resolve to the identity matrix if the samples are presented
a balanced number of times, whereas the weights columns for droppoed samples will never resolve at all. With imbalanced
shards, we should expect to see nearly an identity matrix.
"""

from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from composer.datasets.webdataset_utils import load_webdataset


def parse_args():
    args = ArgumentParser()
    args.add_argument('--out', type=str, default='')
    args.add_argument('--size', type=int, default=1024)
    args.add_argument('--remote', type=str, default='/datasets/wds_synth_1024/')
    args.add_argument('--name', type=str, default='wds_synth_1024')
    args.add_argument('--cache_dir', type=str, default='/tmp/webdataset_cache')
    args.add_argument('--cache_verbose', type=int, default=0)
    args.add_argument('--shuffle', type=int, default=1)
    args.add_argument('--shuffle_buffer', type=int, default=64)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--train_drop_last', type=int, default=1)
    args.add_argument('--val_drop_last', type=int, default=0)
    args.add_argument('--epochs', type=int, default=1000)
    return args.parse_args()


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dim) / dim)

    def forward(self, x):
        return x @ self.weight


def main(args):
    transform = transforms.Compose([
        lambda b: np.frombuffer(b, np.float32).copy(),
        torch.Tensor,
    ])
    preprocess = lambda dataset: dataset.to_tuple('x', 'y').map_tuple(transform, transform)
    n_devices = 1
    workers_per_device = 8
    load_wds = lambda split: load_webdataset(args.remote, args.name, split, args.cache_dir, bool(args.cache_verbose),
                                             bool(args.shuffle), args.shuffle_buffer, preprocess, n_devices,
                                             workers_per_device, args.batch_size, bool(args.train_drop_last))

    train_dataset = load_wds('train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=None,
                                  drop_last=bool(args.train_drop_last))

    val_dataset = load_wds('val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=None,
                                drop_last=bool(args.val_drop_last))

    model = Model(args.size)
    opt = Adam(model.parameters())

    for epoch in range(args.epochs):
        print('Epoch', epoch)
        for x, y_true in train_dataloader:
            opt.zero_grad()
            y_pred = model(x)
            e = F.mse_loss(y_pred, y_true)
            e.backward()
            opt.step()
        for x, y_true in val_dataloader:
            with torch.no_grad():
                y_pred = model(x)
                e = F.mse_loss(y_pred, y_true)

    if args.out:
        w = model.weight.detach().cpu().numpy()
        w.tofile(args.out)


if __name__ == '__main__':
    main(parse_args())
