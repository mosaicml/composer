import pytest
import pdb 
from typing import Dict
from composer.core.state import State
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.lm_datasets import LMDatasetHparams
from composer.models import BERTHparams, GPT2Hparams
from tests.common.models import generate_dummy_model_config
from tests.datasets import test_synthetic_lm_data
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder

def make_dataset_configs():
    lm_dataset_configs = [config[0] for config in test_synthetic_lm_data.generate_parameter_configs( ['num_samples', 'chars_per_sample', 'column_names', 'tokenizer_family'])]
    for config in lm_dataset_configs:
        config['drop_last'] = False
        config['use_masked_lm'] = config['tokenizer_family'] == 'bert'
        if config['use_masked_lm']:
            config['mlm_probability'] = 0.15
    return lm_dataset_configs

def make_lm_tokenizer(config: Dict):
    dataset = synthetic_hf_dataset_builder(num_samples=config['num_samples'],
                                        chars_per_sample=config['chars_per_sample'],
                                        column_names=config['column_names'])
    tokenizer = generate_synthetic_tokenizer(config['tokenizer_family'], dataset)
    return tokenizer

def make_dummy_lm(model_name: str, max_position_embeddings, tokenizer):
    pytest.importorskip("transformers")
    if model_name == 'gpt2':
        class_name = GPT2Hparams
    elif model_name == 'bert':
        class_name = BERTHparams
    model_config = generate_dummy_model_config(class_name, tokenizer)
    if model_name == 'bert':
        model_config['num_labels'] = model_config['vocab_size']
        model_config['max_position_embeddings'] = max_position_embeddings
    model = class_name(model_config=model_config).initialize_object()
    return model

def synthetic_to_dataloader(dataset_config):
    """
    if tokenizer.pad_token_id is None:
        data_collator = transformers.default_data_collator
    else:
        print('using datacollecter for language modeling')
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                        mlm=dataset_config['use_masked_lm'],
                                                                        mlm_probability=dataset_config['mlm_probability'])
    sampler = dist.get_sampler(
            cast(Dataset, dataset),  # HF datasets do not subclass torch datasets, so this cast is needed
            drop_last=dataset_config['drop_last'],
            shuffle=True)
    """
    dataloader = LMDatasetHparams(use_synthetic=True, tokenizer_name=dataset_config['tokenizer_family'], use_masked_lm=dataset_config['use_masked_lm'], split='train', max_seq_length=dataset_config["chars_per_sample"])
    dataloader = dataloader.initialize_object(batch_size=dataset_config['num_samples'], dataloader_hparams=DataLoaderHparams())
    return dataloader

def minimal_lm_state(model, dataloader, rank_zero_seed=0):
    """Most minimally defined state possible.

    Tests should configure the state for their specific needs.
    """
    state = State(
        model=model,
        rank_zero_seed=rank_zero_seed,
        dataloader=dataloader,
        dataloader_label="train",
        max_duration='1ep',
    )
    state.batch = next(iter(state.dataloader)).data
    return state

@pytest.mark.parametrize("config", make_dataset_configs())
def test_minimal_lm_state(config):
    tokenizer = make_lm_tokenizer(config)
    lm = make_dummy_lm(config['tokenizer_family'], config['chars_per_sample'], tokenizer)
    dataloader = synthetic_to_dataloader(config)
    sample =  next(iter(dataloader)).data
    output = lm(sample)
    state = minimal_lm_state(lm, dataloader)
    assert hasattr(state, "batch")
    state_output = state.model(state.batch)
    assert state_output.keys() == output.keys()
    assert state_output.loss.size() == output.loss.size()
    assert state_output.logits.size() == output.logits.size()
    assert state.batch.keys() == sample.keys()
    for key in state.batch.keys():
        assert state.batch[key].size() == sample[key].size()