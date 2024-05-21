"""Script to detect positions of crystals in a folder of drop images using
Mask-R-CNN based object detector.
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

from base.chimp_errors import InputError
import detector.legacy.chimp_utils as utils
from detector.data.coord_generator import PointsMode

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LOGGING_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATE_FMT = "%d-%b-%y %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOGGING_FMT, datefmt=LOGGING_DATE_FMT)


def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        description="CHiMP (Crystal Hits in My Plate) detector"
    )
    parser.add_argument(
        "--MODEL_PATH", required=True, type=str, help="path to model file."
    )
    parser.add_argument(
        "--IMAGE_PATH", required=True, type=str, help="path to image directory."
    )
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--mode", default="SINGLE", type=str)
    parser.add_argument("--preview", default=False, action="store_true")
    parser.add_argument("--masks", default=False, action="store_true")
    parser.add_argument("--echo", default=False, action="store_true")
    return parser


parser = init_argparse()
args = vars(parser.parse_args())
IMAGE_PATH = Path(args["IMAGE_PATH"])
MODEL_PATH = Path(args["MODEL_PATH"])
NUM_CLASSES = args["num_classes"]
MODE = PointsMode[args["mode"]]
PREVIEW_FLAG = args["preview"]
MASKS_FLAG = args["masks"]
ECHO_FLAG = args["echo"]

try:
    utils.check_image_dir_path(IMAGE_PATH)
except InputError as e:
    logging.error(e)
    sys.exit(1)

print("##### CHiMP (Crystal Hits in My Plate) detector #####")
logging.info("Loading libraries...")
from detector.model.operations.detector_predictor import ChimpDetectorPredictor
from detector.data.coord_generator import ChimpXtalCoordGenerator
from detector.data.mask_saver import ChimpXtalMaskSaver

##### Run the script #####
image_list = list(IMAGE_PATH.glob("*"))
try:
    detector = ChimpDetectorPredictor(MODEL_PATH, image_list, NUM_CLASSES)
except InputError as e:
    logging.error(e)
    sys.exit(1)
cwd = Path.cwd()
results_dir = cwd / "detector_output"
logging.info(f"Making directory for detector output: {results_dir}")
results_dir.mkdir(exist_ok=True)
if MASKS_FLAG:
    mask_saver = ChimpXtalMaskSaver(detector, results_dir)
    mask_saver.extract_masks()
else:
    coord_generator = ChimpXtalCoordGenerator(
        detector, points_mode=MODE, extract_echo=ECHO_FLAG
    )
    coord_generator.extract_coordinates()
    coord_generator.save_output_csv(results_dir / "detector_positions.csv")
    if PREVIEW_FLAG:
        logging.info("Output of preview images requested:")
        im_out_dir = results_dir / "preview_images"
        logging.info(f"Making directory for preview image output: {im_out_dir}")
        im_out_dir.mkdir(exist_ok=True)
        coord_generator.save_preview_images(im_out_dir)
