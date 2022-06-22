from composer.trainer import TrainerHparams


def main() -> None:
    # Create a TrainerHparams object from the input file
    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv

    # Create a Trainer object from the TrainerHparams
    trainer = hparams.initialize_object()

    # Call fit() to train the model for a duration
    # equal to the specified max_duration in the input yaml file
    trainer.fit()


if __name__ == '__main__':
    main()
