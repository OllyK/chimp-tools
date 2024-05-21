import logging
from enum import Enum
from itertools import chain
import csv
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndimage, signal
from matplotlib import pyplot as plt
from numpy.random import default_rng
from skimage import io as skio
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2gray
from skimage.transform import rescale
from typing import Union

import detector.legacy.detector_utils as utils

MICRONS_PER_PIXEL_X = 2.837
MICRONS_PER_PIXEL_Y = 2.837
FILE_DIR = Path(__file__).resolve()
ECHO_CONFIG_PATH = FILE_DIR.parents[1] / "config" / "echolocator_config.yaml"
IM_SCALE_FACTOR = 0.125


class PointsMode(Enum):
    RANDOM = 1
    REGULAR = 2
    SINGLE = 3


class ChimpXtalCoordGenerator:
    def __init__(
        self,
        detector,
        prob_threshold=0.6,
        mask_threshold=0.5,
        extract_echo=False,
        points_mode=PointsMode.SINGLE,
        points_min_dist=40,
        edge_min_dist=10,
        grid_spacing=30,
    ):
        self.detector = detector
        self.prob_threshold = prob_threshold
        self.mask_threshold = mask_threshold
        self.extract_echo = extract_echo
        self.points_mode = points_mode
        self.points_min_dist = points_min_dist
        self.edge_min_dist = edge_min_dist
        self.grid_spacing = grid_spacing
        self.combined_coords_list = []
        # Specify the client type and remote endpoint.
        if self.extract_echo and self.detector.num_classes < 3:
            logging.error(
                "Can not extract Echo coordinate without both drop and crystal labels!"
            )
            sys.exit(1)

    def save_output_csv(self, output_path: Union[Path, str]) -> None:
        logging.info(f"Saving coordinates out to {output_path}")
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = ["Filepath", "Coords", "Crystals"]
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for output_dict in self.combined_coords_list:
                if len(output_dict["xtal_coordinates"]) > 0:
                    crystals = True
                else:
                    crystals = False
                if not self.extract_echo:
                    # unpack lists
                    coords = list(chain.from_iterable(output_dict["xtal_coordinates"]))
                    # convert to tuples
                    coords = [tuple(x.tolist()) for x in coords]
                else:
                    if output_dict["drop_detected"] is True:
                        coords = output_dict["echo_coordinate"]
                        orig_im_shape = output_dict["original_image_shape"]
                        coords = utils.scale_indices_to_real_space(
                            coords,
                            (MICRONS_PER_PIXEL_Y, MICRONS_PER_PIXEL_X),
                            orig_im_shape,
                        )
                        coords = tuple(coords[0])
                fp = output_dict["image_path"]
                writer.writerow([fp, coords, crystals])


    def save_preview_images(self, output_dir: Path, scale: int = 0.5) -> None:
        scale_factors = (scale, scale)
        for output_dict in self.combined_coords_list:
            if not self.extract_echo or (output_dict["drop_detected"] is True):
                # Load in image
                img_path = Path(output_dict["image_path"])
                img = utils.load_and_rescale_image(img_path, scale)
                plt.imshow(img, cmap="gray")
                coords_list = utils.scale_indices(
                    output_dict["xtal_coordinates"], scale_factors
                )
                if len(coords_list) > 0:  # Only output plots if something found
                    for coords in coords_list:
                        plt.scatter(
                            y=[coord[0] for coord in coords],
                            x=[coord[1] for coord in coords],
                            marker="+",
                            linewidths=1,
                            s=100,
                        )
                if self.extract_echo:
                    echo_coord = utils.scale_indices(
                        output_dict["echo_coordinate"], scale_factors
                    )
                    plt.plot(
                        echo_coord[0][1], echo_coord[0][0], "ro"
                    )  # is plot(x, y) not plot(y, x)
                    well_centroid = output_dict["well_centroid"]
                    if well_centroid is not None:
                        well_centroid = utils.scale_indices(
                            [well_centroid], scale_factors
                        )
                        plt.plot(well_centroid[0][1], well_centroid[0][0], "bo")

                fig_out_path = output_dir / f"{img_path.stem}_preview.png"
                logging.info(f"Saving figure out to {fig_out_path}")
                plt.axis("off")
                plt.savefig(fig_out_path, bbox_inches="tight")
                plt.close()

    def read_and_resize_image(self, im_path: Union[Path, str]) -> np.array:
        im_array = skio.imread(im_path)
        im_array = rescale(rgb2gray(im_array), IM_SCALE_FACTOR)
        im_array = img_as_ubyte(im_array)
        return im_array

    def extract_coordinates(self) -> None:
        drop_mask = None
        logging.info("Extracting coordinates...")
        for prediction, im_shape_path_tuple in self.detector.detector_output:
            output_dict = utils.create_detector_output_dict(
                prediction, im_shape_path_tuple, prob_threshold=self.prob_threshold
            )
            if self.detector.num_classes == 3:  # Drop and crystal detector
                mask_index = output_dict["mask_index"]
                drop_indices = np.where(output_dict["labels"] == 1)[0]
                xtal_indices = np.where(output_dict["labels"] == 2)[0]
                drop_indices_thresh = np.intersect1d(mask_index, drop_indices)
                xtal_indices_thresh = np.intersect1d(mask_index, xtal_indices)
                num_drops = len(drop_indices_thresh)
                num_xtals = len(xtal_indices_thresh)
                logging.info(
                    f"{num_drops} drops, {num_xtals} crystals over prob threshold"
                )
                if self.extract_echo:
                    if num_drops == 0:
                        # Append the list and continue to next item
                        logging.info("No drop detected over probabilty threshold!")
                        # drop_index = drop_indices[0] # Take drop coord under threshold - is this necessary?
                        self.combined_coords_list.append(output_dict)
                        continue
                    else:
                        output_dict["drop_detected"] = True
                        drop_index = drop_indices_thresh[0]
                    drop_mask = prediction[0]["masks"][drop_index, 0]  # tensor mask
                    drop_mask = utils.threshold_mask(
                        drop_mask, mask_threshold=self.mask_threshold
                    )  # numpy mask
            else:
                xtal_indices_thresh = output_dict["mask_index"]
            # If there is an empty drop, calculate centre of mass and continue
            if len(xtal_indices_thresh) == 0:
                logging.info("No crystal detected over probability threshold!")
                if drop_mask is not None:
                    drop_com = ndimage.center_of_mass(drop_mask)
                    drop_com = [list(drop_com)]
                    new_im_shape = drop_mask.shape
                    scale_factors = utils.calculate_scale_factors(
                        output_dict["original_image_shape"], new_im_shape
                    )
                    drop_com = utils.scale_indices(drop_com, scale_factors)
                    output_dict["echo_coordinate"] = drop_com
                    self.combined_coords_list.append(output_dict)
                continue
            for i in xtal_indices_thresh:
                xtal_mask = prediction[0]["masks"][i, 0]  # tensor mask
                xtal_mask = utils.threshold_mask(
                    xtal_mask, mask_threshold=self.mask_threshold
                )  # numpy mask
                if self.extract_echo:
                    # Subtract Xtal mask from drop mask
                    drop_mask = drop_mask - xtal_mask.clip(None, drop_mask)
                new_im_shape = xtal_mask.shape
                scale_factors = utils.calculate_scale_factors(
                    output_dict["original_image_shape"], new_im_shape
                )
                obj_indices = np.where(xtal_mask == 1)  # All pixels in segmented object
                obj_indices = self.sparsify_indices(obj_indices, xtal_mask)
                # Scale coordinates to fit original image size
                obj_indices = utils.scale_indices(obj_indices, scale_factors)
                output_dict["xtal_coordinates"].append(obj_indices)
            if self.extract_echo:
                logging.info("Extracting Echo coordinate from distance transform.")
                dist_trans = ndimage.distance_transform_edt(drop_mask)
                echo_coord = list(
                    np.unravel_index(np.argmax(dist_trans), dist_trans.shape)
                )
                echo_coord = [list(echo_coord)]
                # Scale coordinate to fit original image size
                echo_coord = utils.scale_indices(echo_coord, scale_factors)
                output_dict["echo_coordinate"] = echo_coord
            self.combined_coords_list.append(output_dict)
        return self.combined_coords_list

    def sparsify_indices(self, obj_indices: np.array, mask: np.array) -> list:
        coords_list_in = list(zip(obj_indices[0], obj_indices[1]))
        edge_coords = utils.calculate_edges(mask)
        area = len(coords_list_in)  # pixel area of object
        c_of_m = utils.calculate_centre_of_mass(mask)
        if area < 100 or self.points_mode == PointsMode.SINGLE:
            return [c_of_m]
        elif self.points_mode == PointsMode.RANDOM:
            return self.generate_random_coords_list(
                area, c_of_m, coords_list_in, edge_coords
            )
        elif self.points_mode == PointsMode.REGULAR:
            return self.generate_regular_coords_list(mask, c_of_m, edge_coords)

    def generate_regular_coords_list(
        self, mask: np.array, c_of_m: np.array, edge_coords: np.array
    ) -> list:
        mask_y, mask_x = mask.shape
        com_y, com_x = c_of_m
        grid = np.meshgrid(
            np.arange(com_y % self.grid_spacing, mask_y, self.grid_spacing),
            np.arange(com_x % self.grid_spacing, mask_x, self.grid_spacing),
        )
        # TODO: Replace this with something cleverer
        mask_copy = mask.copy()
        mask_copy[tuple(grid)] = 2
        masked_arr = np.ma.masked_array(mask_copy, ~mask.astype(bool))
        grid_indices = np.where(masked_arr.filled([0]) == 2)
        grid_coords = list(zip(grid_indices[0], grid_indices[1]))
        coords_list_out = []
        for g_coord in grid_coords:
            edge_dist_arr = np.linalg.norm(g_coord - edge_coords, axis=1)
            if any(edge_dist_arr < self.edge_min_dist):
                pass
            else:
                coords_list_out.append(g_coord)
        if len(coords_list_out) == 0:
            coords_list_out.append(c_of_m)
        return coords_list_out

    def generate_random_coords_list(
        self, area: int, c_of_m: np.array, coords_list_in: list, edge_coords: np.array
    ) -> list:
        coords_list_out = [c_of_m]
        rng = default_rng()
        size = (
            int(area / 20) if int(area / 20) else 1
        )  # limit the number of points to try
        random_selection = rng.choice(area, size=size, replace=False)
        for rand_int in random_selection:
            rand_coord = np.array(coords_list_in[rand_int])
            # Calculate the distances to all other points in the output array
            dist_arr = np.linalg.norm(rand_coord - coords_list_out, axis=1)
            # Calculate distance to points along mask edge
            edge_dist_arr = np.linalg.norm(rand_coord - edge_coords, axis=1)
            if any(dist_arr < self.points_min_dist) or any(
                edge_dist_arr < self.edge_min_dist
            ):
                pass
            else:
                coords_list_out.append(rand_coord)
        return coords_list_out
