import logging
from pathlib import Path
import scipy.ndimage as ndimage

import numpy as np
import cv2
import torch
from typing import Tuple, List
import detector.legacy.detector_config as cfg


def load_and_rescale_image(img_path: Path, scale: float) -> np.ndarray:
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, None, fx=scale, fy=scale)


def threshold_mask(mask: torch.tensor, mask_threshold: float = 0.5) -> np.ndarray:
    mask[mask > mask_threshold] = 1
    mask[mask <= mask_threshold] = 0
    return np.squeeze(mask.numpy().astype(np.uint8))


def calculate_scale_factors(
    original_im_shape: Tuple[int, int], new_im_shape: Tuple[int, int]
) -> Tuple[float, float]:
    return (
        original_im_shape[0] / new_im_shape[0],
        original_im_shape[1] / new_im_shape[1],
    )


def calculate_centre_of_mass(mask: np.ndarray) -> np.ndarray:
    c_of_m = ndimage.center_of_mass(mask)
    return np.rint(c_of_m).astype(int)


def calculate_edges(mask: np.ndarray) -> np.ndarray:
    convolved = ndimage.convolve(mask, cfg.EDGE_KERNEL, mode="constant")
    indices = np.where(convolved > 1)
    return np.array(list(zip(indices[0], indices[1])))


def scale_indices(
    obj_indices: List[List[int]], scale_factors: Tuple[float, float]
) -> List[np.ndarray]:
    if len(obj_indices) == 0:
        return obj_indices
    return [np.rint(x * np.array(scale_factors)).astype(int) for x in obj_indices]


def scale_indices_to_real_space(
    obj_indices: List[List[int]], scale_factors: Tuple[float, float], orig_im_shape
) -> List[np.ndarray]:
    if len(obj_indices) == 0:
        return list(np.array(0))
    y, x = orig_im_shape
    centres = np.array([y // 2, x // 2])
    offsets = obj_indices - centres
    return [x * np.array(scale_factors) for x in offsets]


def calculate_realspace_offset(
    echo_coords: np.array(int),
    well_centre: np.array(int),
    scale_factors: np.array(float),
) -> np.array(int):
    return np.rint((echo_coords - well_centre) * scale_factors)


def create_detector_output_dict(
    prediction, im_shape_path_tuple, prob_threshold=0.6
) -> dict:
    original_im_shape, image_path = im_shape_path_tuple
    probs = prediction[0]["scores"].numpy()
    labels = prediction[0]["labels"].cpu().numpy()
    mask_index = np.where(probs >= prob_threshold)[0]
    logging.info(f"{image_path.name} - Number of objects found: " f"{len(mask_index)}")
    output_dict = {
        "image_path": str(image_path),
        "mask_index": mask_index,
        "masks": [],
        "probs": [],
        "labels": labels,
        "bounding_boxes": [],
        "xtal_coordinates": [],
        "well_centroid": None,
        "echo_coordinate": [],
        "real_space_offset": None,
        "original_image_shape": original_im_shape,
        "drop_detected": False,
    }
    return output_dict
