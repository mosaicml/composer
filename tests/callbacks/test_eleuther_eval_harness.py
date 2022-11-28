# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import DataLoader

from composer.callbacks import EleutherEvalHarness
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledSGDW
from composer.trainer import Trainer
from tests.common.datasets import RandomCausalLMDataset
from composer.models.gpt2 import create_gpt2
import transformers

def test_eleuther_eval_harness():
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    # Construct the callback
    eval_callback = EleutherEvalHarness(tokenizer=tokenizer, task_list=["lambada"], num_fewshot=[0], subsample_size=8)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger

    # Construct the trainer and train
    model = create_gpt2(use_pretrained=True, pretrained_model_name="EleutherAI/gpt-neo-125M")
    optimizer = DecoupledSGDW(model.parameters(), lr=1e-12)
    trainer = Trainer(
        model=model,
        callbacks=eval_callback,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomCausalLMDataset(size=1, vocab_size=tokenizer.vocab_size, sequence_length=2048)),
        eval_dataloader=DataLoader(RandomCausalLMDataset(size=1, vocab_size=tokenizer.vocab_size, sequence_length=2048)),
        max_duration='1ep',
        eval_interval='1ep',
        optimizers=optimizer
    )
    trainer.fit()

