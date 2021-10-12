"""
Runs the mosaic trainer with the provided yaml file. e.g.

> python examples/run_mosaic_trainer.py -f mosaicml/models/classify_mnist/hparams.yaml

"""
import argparse
import logging

from composer.trainer import Trainer, TrainerHparams

logging.basicConfig()
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
logging.getLogger("composer").setLevel(logging.INFO)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-f',
    '--file',
    default=None,
    type=str,
    help='Path to hparams YAML file.',
)
parser.add_argument(
    '--create-template',
    action="store_true",
    dest="template",
    default=False,
    help='Template a new hparams YAML file',
)
parser.add_argument(
    '--datadir',
    default=None,
    help='set the datadir for both train and eval datasets',
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.template:
        TrainerHparams.interactive_template()
        exit(0)
    if args.file is None:
        raise Exception("Please pass a YAML file to create Hparams from")

    hparams = TrainerHparams.create(args.file)
    assert isinstance(hparams, TrainerHparams)

    if args.datadir is not None:
        if not hasattr(hparams.train_dataset, 'datadir') or \
            not hasattr(hparams.val_dataset, 'datadir'):
            raise ValueError('To set with --datadir, both the train and val '
                             'dataset must have the datadir attribute.')
        setattr(hparams.train_dataset, 'datadir', args.datadir)
        setattr(hparams.val_dataset, 'datadir', args.datadir)
        logger.info(f'Set dataset dirs in hparams to: {args.datadir}')

    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()
