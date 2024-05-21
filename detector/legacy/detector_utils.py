import logging
from pathlib import Path
import numpy as np
import cv2
import torch
from typing import Tuple, List

import scipy.ndimage as ndimage

import detector.legacy.detector_config as cfg


def load_and_rescale_image(img_path: Path, scale: float) -> np.ndarray:
    """
    Load an image from the given path and rescale it.

    Args:
        img_path (Path): The path to the image file.
        scale (float): The scale factor to resize the image.

    Returns:
        np.ndarray: The rescaled image as a NumPy array.
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, None, fx=scale, fy=scale)


def threshold_mask(mask: torch.tensor, mask_threshold: float = 0.5) -> np.ndarray:
    """
    Apply a threshold to the mask.

    Args:
        mask (torch.tensor): The mask tensor.
        mask_threshold (float, optional): The threshold value. Defaults to 0.5.

    Returns:
        np.ndarray: The thresholded mask as a NumPy array.
    """
    mask[mask > mask_threshold] = 1
    mask[mask <= mask_threshold] = 0
    return np.squeeze(mask.numpy().astype(np.uint8))


def calculate_scale_factors(
    original_im_shape: Tuple[int, int], new_im_shape: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Calculates the scale factors between the original image shape and the new image shape.

    Args:
        original_im_shape (Tuple[int, int]): The shape of the original image (height, width).
        new_im_shape (Tuple[int, int]): The shape of the new image (height, width).

    Returns:
        Tuple[float, float]: The scale factors between the original image shape and the new image shape.
    """
    return (
        original_im_shape[0] / new_im_shape[0],
        original_im_shape[1] / new_im_shape[1],
    )


def calculate_centre_of_mass(mask: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass of the mask.

    Args:
        mask (np.ndarray): The mask as a NumPy array.

    Returns:
        np.ndarray: The center of mass coordinates as a NumPy array.
    """
    c_of_m = ndimage.center_of_mass(mask)
    return np.rint(c_of_m).astype(int)


def calculate_edges(mask: np.ndarray) -> np.ndarray:
    """
    Calculate the edges of the mask.

    Args:
        mask (np.ndarray): The mask as a NumPy array.

    Returns:
        np.ndarray: The edge coordinates as a NumPy array.
    """
    convolved = ndimage.convolve(mask, cfg.EDGE_KERNEL, mode="constant")
    indices = np.where(convolved > 1)
    return np.array(list(zip(indices[0], indices[1])))


def scale_indices(
    obj_indices: List[List[int]], scale_factors: Tuple[float, float]
) -> List[np.ndarray]:
    """
    Scale the object indices based on the scale factors.

    Args:
        obj_indices (List[List[int]]): The object indices.
        scale_factors (Tuple[float, float]): The scale factors.

    Returns:
        List[np.ndarray]: The scaled object indices.
    """
    if len(obj_indices) == 0:
        return obj_indices
    return [np.rint(x * np.array(scale_factors)).astype(int) for x in obj_indices]


def scale_indices_to_real_space(
    obj_indices: List[List[int]], scale_factors: Tuple[float, float], orig_im_shape
) -> List[np.ndarray]:
    """
    Scale the object indices to real space based on the scale factors and original image shape.

    Args:
        obj_indices (List[List[int]]): The object indices.
        scale_factors (Tuple[float, float]): The scale factors.
        orig_im_shape: The original image shape.

    Returns:
        List[np.ndarray]: The scaled object indices in real space.
    """
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
    """
    Calculate the real space offset.

    Args:
        echo_coords (np.array(int)): The echo coordinates.
        well_centre (np.array(int)): The well centre coordinates.
        scale_factors (np.array(float)): The scale factors.

    Returns:
        np.array(int): The real space offset.
    """
    return np.rint((echo_coords - well_centre) * scale_factors)


def create_detector_output_dict(
    prediction, im_shape_path_tuple, prob_threshold=0.6
) -> dict:
    """
    Create a dictionary containing the output of the detector.

    Args:
        prediction (Tensor): The prediction output from the detector model.
        im_shape_path_tuple (tuple): A tuple containing the original image shape and the image path.
        prob_threshold (float, optional): The probability threshold for object detection. Defaults to 0.6.

    Returns:
        dict: A dictionary containing the following keys:
            - image_path (str): The path of the image.
            - mask_index (ndarray): The indices of the detected objects.
            - masks (list): The masks of the detected objects.
            - probs (list): The probabilities of the detected objects.
            - labels (ndarray): The labels of the detected objects.
            - bounding_boxes (list): The bounding boxes of the detected objects.
            - xtal_coordinates (list): The crystal coordinates of the detected objects.
            - well_centroid (None): The centroid of the well.
            - echo_coordinate (list): The echo coordinates of the detected objects.
            - real_space_offset (None): The real space offset.
            - original_image_shape (tuple): The original shape of the image.
            - drop_detected (bool): Indicates whether a drop was detected or not.
    """
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
