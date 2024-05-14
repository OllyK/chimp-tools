#!/usr/bin/env python

from datetime import date
import logging
import click
from pathlib import Path


import utilities.config as cfg
from classifier.model import ImageFolderClassifier


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to classification model file.",
)
@click.option(
    "--image_dir",
    required=True,
    help="Path to directory containing images for classification.",
)
@click.option(
    "--img_size",
    default=384,
    help="Image size to use for inference.",
)
def predict_folder(model_path, image_dir, img_size):
    logging.basicConfig(
        level=logging.INFO,
        format=cfg.LOGGING_FMT,
        datefmt=cfg.LOGGING_DATE_FMT,
        handlers=[
            logging.FileHandler(f"{date.today()}_predict_classifier.log"),
            logging.StreamHandler(),
        ],
    )
    image_dir = Path(image_dir)
    model_path = Path(model_path)
    image_list = list(image_dir.glob("*"))
    output_dir = Path.cwd()
    predictor = ImageFolderClassifier(
        model_path, image_list, img_size, device=cfg.PREDICTION_DEVICE
    )
    predictor.run(output_dir)



if __name__ == "__main__":
    predict_folder()
