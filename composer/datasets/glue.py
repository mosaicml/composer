import logging
from multiprocessing import cpu_count

import transformers
import yahp as hp
from lm_datasets import _split_dict_fn

from composer.datasets.hparams import DataloaderSpec, DatasetHparams

log = logging.getLogger(__name__)


class SST2(DatasetHparams):
    tokenizer_name: str = hp.required("The name of the tokenizer to preprocess text with.")
    split: str = hp.required("Whether to use 'train', 'validation' or 'test' split.")
    use_masked_lm: bool = hp.required("Whether the dataset shoud be encoded with masked language modeling or not.")
    mlm_probability: float = hp.optional("If using masked language modeling, the probability to mask tokens with.",
                                         default=0.15)
    max_seq_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    val_sequence_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the validation dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def validate(self):
        raise NotImplementedError("Validation still needs to be implemented.")

    def initialize_object(self) -> DataloaderSpec:
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.dataset = datasets.load_dataset("glue", "sst2", split=self.split)

        log.info("Starting tokenization step...")
        n_cpus = cpu_count()
        text_column_name = "sentence"

        def tokenize_function(inp):
            return self.tokenizer(
                inp[text_column_name],
                padding=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=True,
            )

        self.dataset[self.split] = self.dataset[self.split].map(
            tokenize_function,
            batched=True,
            num_proc=n_cpus,
            batch_size=1000,
            # remove_columns=column_names,  # TODO: consider if this is necessary
            new_fingerprint=f"sst2-tokenization-{self.split}",
            load_from_cache_file=True,
        )

        self.data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                                          mlm=self.use_masked_lm,
                                                                          mlm_probability=self.mlm_probability)

        return DataloaderSpec(
            dataset=self.dataset,
            collate_fn=self.data_collator,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            split_fn=_split_dict_fn,
        )
