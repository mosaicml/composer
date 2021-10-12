import logging

import composer
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    hparams = TrainerHparams.create()
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


if __name__ == "__main__":
    main()
