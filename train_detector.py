#!/usr/bin/env python

from datetime import date
import logging
from pathlib import Path

import click

import utilities.config as cfg
from detector.model import MaskRCNNTrainer
from utilities.base_data_utils import get_detector_train_settings


@click.command()
@click.option(
    "--settings_file",
    default=cfg.DET_TRAIN_SETTINGS_FN,
    show_default=True,
    help="Path to settings file.",
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
@click.argument("data_dir_path")
def train_detector(
    data_dir_path,
    settings_file,
    finetune,
    imbalanced,
    fix_seed,
    reload_optimizer,
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
    settings = get_detector_train_settings(settings_file)
    num_epochs = settings.num_epochs
    # Create Trainer
    trainer = MaskRCNNTrainer(
        data_dir_path,
        settings,
        finetune_path=finetune,
        fix_seed=fix_seed,
    )
    second_round = finetune is not None
    model_fn = f"{date.today()}_{settings.model_output_fn}.pytorch"
    model_out = Path(Path.cwd(), model_fn)
    trainer.train_model(model_out, num_epochs, settings.patience)
    trainer.output_loss_fig(model_out)
        


if __name__ == "__main__":
    train_detector()
