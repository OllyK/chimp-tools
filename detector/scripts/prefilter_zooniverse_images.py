""" Script to process images using CLAHE and create a manifest file """
import click
import cv2
from pathlib import Path
import pandas as pd


# Function to adjust image contrast using CLAHE
def contrast_adjust(img, gridsize):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Function to create a manifest of processed images
def create_manifest_from_dir(dir_path, prefix_text):
    columns = ['filename']
    csv = list(dir_path.glob("*.jpg"))
    df = pd.DataFrame(csv, columns=columns)
    df.index.name = 'id'
    df.to_csv(dir_path / f"{prefix_text}_manifest.csv")

@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_dir', type=click.Path())
@click.option('--gridsize', default=12, help='Grid size for CLAHE')
@click.option('--prefix', default='processed', help='Prefix for the output manifest file')
def process_images(input_dir, output_dir, gridsize, prefix):
    """ Process images in the INPUT_DIR using CLAHE and save the results to OUTPUT_DIR """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)

    # Process images
    for img_path in input_path.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        adjusted = contrast_adjust(img, gridsize)
        cv2.imwrite(str(output_path / img_path.name), adjusted, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    # Create a manifest file in the output directory
    create_manifest_from_dir(output_path, prefix)

if __name__ == '__main__':
    process_images()
