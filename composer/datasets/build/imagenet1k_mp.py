from argparse import ArgumentParser
from glob import glob
from io import BytesIO
from multiprocessing import Process, Queue
import os
from PIL import Image
from random import shuffle
from tqdm import tqdm
from webdataset import ShardWriter


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in_root', type=str, required=True)
    x.add_argument('--out_root', type=str, required=True)
    x.add_argument('--train_shards', type=int, default=128)
    x.add_argument('--val_shards', type=int, default=16)
    x.add_argument('--procs', type=int, default=8)
    return x.parse_args()


def find_samples(in_root, split):
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


def produce(todo_q, data_q):
    while True:
        item = todo_q.get()
        if item is None:
            break
        filename, klass = item
        im = Image.open(filename)
        im = im.convert('RGB')
        data = BytesIO()
        im.save(data, 'JPEG')
        data = data.getvalue()
        data_q.put((data, klass))


def consume(data_q, pattern, n_samples, n_shards):
    dirname = os.path.dirname(pattern)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    shard_size = n_samples // n_shards
    writer = ShardWriter(pattern, maxcount=shard_size)
    pbar = tqdm(total=n_samples, leave=False)
    sample_idx = 0
    while True:
        item = data_q.get()
        if item is None:
            break
        data, klass = item
        x = {
            '__key__': '%05d' % sample_idx,
            'jpg': data,
            'cls': klass,
        }
        writer.write(x)
        pbar.update(1)
        sample_idx += 1
    writer.close()


def process_split(in_root, out_root, split, n_shards, n_procs):
    # Find and shuffle list of dataset samples.
    todos = find_samples(in_root, split)

    # Populate the todo queue and add stop tokens.
    todo_q = Queue()
    for todo in todos:
        todo_q.put(todo)
    for i in range(n_procs):
        todo_q.put(None)

    # Start the producers, which take filenames off the todo queue and put
    # JPEG encoded data on the data queue.
    data_q = Queue()
    producers = [Process(target=produce, args=(todo_q, data_q)) for i in range(n_procs)]
    for p in producers:
        p.start()

    # Start the consumer, which writes the samples it pops off the data queue.
    pattern = os.path.join(out_root, '%s_%%05d.tar' % split)
    n_samples = len(todos)
    consumer = Process(target=consume, args=(data_q, pattern, n_samples, n_shards))
    consumer.start()

    # Join the producers as they complete.
    for p in producers:
        p.join()

    # Join the consumer when it completes.
    data_q.put(None)
    consumer.join()


def main(args):
    process_split(args.in_root, args.out_root, 'train', args.train_shards, args.procs)
    process_split(args.in_root, args.out_root, 'val', args.val_shards, args.procs)


if __name__ == '__main__':
    main(parse_args())
