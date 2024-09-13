# Description: This script processes a CSV file containing Zooniverse annotations and generates image masks from the annotations.
import click
import pandas as pd
from pathlib import Path
import numpy as np
import PIL.Image
import PIL.ImageDraw
import json

def get_masks_from_points(item_list, height, width):
    mask_list = []
    empty_im = np.zeros((height, width)).astype(np.uint8)
    for masked_item in item_list:
        points_lst = masked_item['points'] # Get list of dictionaries
        points_lst = [(x["x"], x["y"]) for x in points_lst] # Convert dicts to tuples
        mask_im = PIL.Image.fromarray(empty_im)
        if len(points_lst) > 2:
            draw = PIL.ImageDraw.Draw(mask_im)
            draw.polygon(xy=points_lst, outline=1, fill=1)
            mask_list.append(np.array(mask_im, dtype=bool))
    return mask_list

def save_masks_from_df_row(row, output_dir):
    im_filename = row['filepath']
    width, height = PIL.Image.open(im_filename).size
    drop_masks = get_masks_from_points(row["drop_task"], height, width)
    xtal_masks = get_masks_from_points(row["xtal_task"], height, width)
    drop_labels = ["drop"] * len(drop_masks)
    xtal_labels = ["crystal"] * len(xtal_masks)
    masks = np.array(drop_masks + xtal_masks, dtype=bool)
    class_labels = np.array(drop_labels + xtal_labels)
    np.savez_compressed(output_dir/f"{im_filename.stem}.npz", class_labels=class_labels, masks=masks)

def setup_dataframe(df):
    # Load in JSON
    df['annotations'] = df['annotations'].apply(json.loads)
    df['subject_data'] = df['subject_data'].apply(json.loads)
    # Extract filepath
    df["filepath"] = df.apply(lambda row: Path(row['subject_data'][str(row['subject_ids'])]['filename']), axis=1)
    # Separate out drop task and crystal task
    df["drop_task"] = df.apply(lambda row: row['annotations'][0]['value'], axis=1)
    df["xtal_task"] = df.apply(lambda row: row['annotations'][1]['value'], axis=1)
    return df

@click.command()
@click.argument('csv_path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('output_dir', type=click.Path())
def process_csv(csv_path, output_dir):
    """ Process a CSV and generate image masks in the OUTPUT_DIR. """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    df = setup_dataframe(df)
    # Iterate over rows and generate masks
    for _, row in df.iterrows():
        save_masks_from_df_row(row, output_path)

if __name__ == '__main__':
    process_csv()
