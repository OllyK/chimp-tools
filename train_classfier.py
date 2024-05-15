#!/usr/bin/env python

from datetime import date
import logging
from pathlib import Path

import click

import utilities.config as cfg
from classifier.model.operations.classifier_trainer import ImageClassifierTrainer
from utilities.base_data_utils import (
    get_classifier_settings,
    get_standard_training_data,
    get_pre_split_data,
)


@click.command()
@click.option(
    "--settings_file",
    default=cfg.CLF_TRAIN_SETTINGS_FN,
    show_default=True,
    help="Path to settings file.",
)
@click.option(
    "--prepend_dir",
    default=None,
    help="Prepend this directory if paths in CSV are relative rather than absolute.",
)
@click.option(
    "--finetune",
    default=None,
    help="Path to existing model to load in for further training.",
)
@click.option(
    "--imbalanced",
    is_flag=True,
    default=False,
    help="Use this option to upsample under-represented classes in imbalanced data.",
)
@click.option(
    "--valid_data",
    default=None,
    help="Path to validation dataset CSV (if data has been split before training).",
)
@click.option(
    "--fix_seed",
    is_flag=True,
    default=False,
    help="Use this option to use the random_seed defined in settings to fix the train/valid split.",
)
@click.option(
    "--reload_optimizer",
    is_flag=True,
    default=False,
    help="Use this option to when adding training an unfrozen model to load in previous optimizer weights.",
)
@click.option(
    "--distributed",
    is_flag=True,
    default=False,
    help="Use this option to perform training distributed across multiple GPUs.",
)
@click.option(
    "--use_accuracy",
    is_flag=True,
    default=False,
    help="Use this option to maximise accuracy metric for saving model and early stopping rather than minimising validation loss.",
)
@click.argument("csv_filepath")
def train_classifier(
    settings_file,
    csv_filepath,
    prepend_dir,
    finetune,
    imbalanced,
    valid_data,
    fix_seed,
    reload_optimizer,
    distributed,
    use_accuracy,
):
    logging.basicConfig(
        level=logging.INFO,
        format=cfg.LOGGING_FMT,
        datefmt=cfg.LOGGING_DATE_FMT,
        handlers=[
            logging.FileHandler(f"{date.today()}_model_training.log"),
            logging.StreamHandler(),
        ],
    )
    settings = get_classifier_settings(settings_file)
    # Standard (non-MARCO training)
    if valid_data is None:
        training_data = get_standard_training_data(csv_filepath, prepend_dir)
        validation_data = None
    else:
        training_data, validation_data = get_pre_split_data(
            csv_filepath,
            valid_data,
            prepend_dir
        )
    # Create Trainer
    trainer = ImageClassifierTrainer(
        training_data,
        settings,
        finetune_path=finetune,
        imbalanced=imbalanced,
        fix_seed=fix_seed,
        marco_validation_data=validation_data,
        distributed=distributed,
    )
    second_round = finetune is not None
    num_cyc_frozen = settings.num_cyc_frozen
    num_cyc_unfrozen = settings.num_cyc_unfrozen
    model_type = settings.model["type"]
    model_fn = f"{date.today()}_{model_type}_{settings.model_output_fn}.pytorch"
    model_out = Path(Path.cwd(), model_fn)
    if num_cyc_frozen > 0:
        frozen = True
        create = True if finetune is None else False
        trainer.train_model(
                model_out, num_cyc_frozen, settings.patience, create=create, frozen=frozen, use_accuracy=use_accuracy,
            )
    if num_cyc_unfrozen > 0 and num_cyc_frozen > 0:
        trainer.train_model(
            model_out,
            num_cyc_unfrozen,
            settings.patience,
            create=False,
            frozen=False,
            second_round=second_round,
            use_accuracy=use_accuracy,
        )
    elif num_cyc_unfrozen > 0 and num_cyc_frozen == 0:
        if finetune is None:
            trainer.train_model(
                model_out,
                num_cyc_unfrozen,
                settings.patience,
                create=True,
                frozen=False,
                use_accuracy=use_accuracy,
            )
        else:
            trainer.train_model(
                model_out,
                num_cyc_unfrozen,
                settings.patience,
                create=False,
                frozen=False,
                reload_optimizer=reload_optimizer,
                use_accuracy=use_accuracy,
            )
    trainer.output_loss_fig(model_out)


if __name__ == "__main__":
    train_classifier()
