import json
import os
from tqdm import tqdm
from webdataset import ShardWriter
from wurlitzer import pipes


def create_webdataset_meta(split_dir, n_samples, n_shards):
    '''Write a WebDataset meta file.'''
    samples_per_shard = n_samples // n_shards
    n_leftover = n_samples % samples_per_shard
    obj = {
        'n_shards': n_shards,
        'samples_per_shard': samples_per_shard,
        'n_leftover': n_leftover,
    }
    filename = os.path.join(split_dir, 'meta.json')
    json.dump(obj, open(filename, 'w'), sort_keys=True)


def create_webdataset(samples, dataset_dir, split, n_samples, n_shards, use_tqdm=1):
    '''Write an entire WebDataset to a local directory, given an iterable of samples.'''
    split_dir = os.path.join(dataset_dir, split)
    os.makedirs(split_dir)
    pattern = os.path.join(split_dir, '%05d.tar')
    samples_per_shard = n_samples // n_shards
    with pipes():
        out = ShardWriter(pattern, maxcount=samples_per_shard)
        out.verbose = 0
    if use_tqdm:
        samples = tqdm(samples, total=n_samples, leave=False)
    for sample in samples:
        out.write(sample)
    out.close()
    create_webdataset_meta(split_dir, n_samples, n_shards)


def download_webdataset_meta(dataset_name, split):
    '''Download a WebDataset meta file from S3.'''
    cmd = f'aws s3 cp s3://mosaicml-internal-dataset-{dataset_name}/{split}/meta.json -'
    return subprocess.run(cmd).stdout


def load_webdataset(dataset_name, split, cache_dir=None, cache_verbose=False):
    '''Initialize a WebDataset pointed at S3 with an optional local cache dir.'''
    if cache_dir:
        split_dir = os.path.join(cache_dir, dataset_name, split)
        meta_file = os.path.join(split_dir, 'meta.json')
        if os.path.exists(meta_file):
            text = open(meta_file).read()
        else:
            text = download_webdataset_meta(dataset_name, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            with open(meta_file, 'w') as out:
                out.write(text)
    else:
        text = download_webdataset_meta(dataset_name, split)
    meta = json.loads(text)
    urls = (f'aws s3 cp s3://mosaicml-internal-dataset-{dataset_name}/split/'
             '{00000..{meta["n_shards"] - 1}}.tar -')
    dataset = WebDataset(urls, cache_dir=cache_dir, cache_verbose=cache_verbose)
    dataset.meta = meta
    return dataset
