# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import textwrap

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from composer.core import DataSpec
from composer.utils import dist, ensure_tuple
from composer.utils.file_helpers import get_file


class InContextLearningLMTaskDataset(Dataset):
    def __init__(
        self,
        dataset_uri: str,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        eos_tok_id: int,
        destination_path: str = 'icl_lm_task.json',
    ):
        get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        self.encoded_dataset = list(
            dataset.map(lambda examples: {
                'continuation': tokenizer(examples['continuation']),
                'context': tokenizer(examples['context']),
            }))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_tok_id = eos_tok_id

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

    def collate_fn(self, data):
        inputs = []
        continuation_indices = []
        for data_pair in data:
            context, continuation = data_pair['context'], data_pair['continuation']

            context_enc = context['input_ids']
            continuation_enc = continuation['input_ids']
            continuation_span = torch.tensor(range(len(context_enc), len(context_enc) + len(continuation_enc)))

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(self.max_seq_len + 1):],
                dtype=torch.long,
            )
            (inplen,) = inp.shape

            # pad length from seq to padding_length
            inp = torch.cat(
                [
                    inp,  # [seq]
                    torch.LongTensor((self.max_seq_len - inplen) * [self.eos_tok_id]),  # [padding_length - seq]
                ],
                dim=0,
            )

            inputs.append(inp)
            continuation_indices.append(continuation_span)

        return {
            'input_ids': torch.stack(inputs),
            'continuation_indices': continuation_indices,
            'eval_forward_handle': self.eval_forward,
            'update_metric_handle': self.update_metric,
            'labels': torch.stack(inputs),
            'tokenizer': self.tokenizer
        }

    def get_num_samples_in_batch(self, batch) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]

        dim0_sizes = []
        if isinstance(batch, (list, tuple)):
            for tensors in batch:
                for t in ensure_tuple(tensors):
                    if not hasattr(t, 'shape'):
                        raise ValueError('Unable to determine the batch size, batch contains'
                                         f'an element of type {type(t)}, which does not have a'
                                         'shape. Please use a DataSpec and provide a'
                                         '`get_num_samples_in_batch(your_batch) -> int` method.')
                    dim0_sizes.append(t.shape[0])
        elif isinstance(batch, dict):
            dim0_sizes = [t.shape[0] for t in batch.values() if isinstance(t, torch.Tensor)]

        if len(set(dim0_sizes)) == 1:
            return dim0_sizes[0]
        else:
            raise NotImplementedError(
                textwrap.dedent(f"""\
                    Cannot determine the batch size, as multiple Tensors of
                    different lengths were found in the batch: sizes in batch: {dim0_sizes}.
                    Please use a DataSpec and specify `get_num_samples_in_batch`."""))

    def eval_forward(self, model, batch):
        while inspect.getfullargspec(model.forward).args == ['self', 'batch']:
            model = model.model

        forward_argspec = inspect.getfullargspec(model.forward).args
        args = {'input_ids': batch['input_ids']}
        if 'key_padding_mask' in forward_argspec:
            # composer gpt uses key padding mask
            args['key_padding_mask'] = ~(batch['input_ids'] == self.eos_tok_id)
        elif 'attention_mask' in forward_argspec:
            # huggingface transformer uses attention_mask
            args['attention_mask'] = ~(batch['input_ids'] == self.eos_tok_id)

        with torch.no_grad():
            res = model(**args)
            if isinstance(res, transformers.modeling_outputs.CausalLMOutputWithPast):
                res = res.logits
            return res

    def update_metric(self, metric, batch, output_logits, labels):
        metric.update(batch, output_logits, labels)


def get_lm_task_dataloader(dataset_uri, tokenizer, global_batch_size, max_seq_len, eos_tok_id):
    dataset = InContextLearningLMTaskDataset(dataset_uri, tokenizer, max_seq_len, eos_tok_id)
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=True)
    batch_size = global_batch_size // dist.get_world_size()
    print(f'Using microbatch size {batch_size}')
    return DataSpec(
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=dataset.collate_fn,
            ),
            device_transforms=None,
            get_num_samples_in_batch=dataset.get_num_samples_in_batch
        )