import logging
from pathlib import Path

import cv2
import numpy as np

import detector.legacy.detector_utils as utils


class ChimpXtalMaskSaver:
    def __init__(self, detector, output_dir, prob_threshold=0.6, mask_threshold=0.5):
        self.detector = detector
        self.output_dir = output_dir
        self.prob_threshold = prob_threshold
        self.mask_threshold = mask_threshold

    def extract_masks(self):
        logging.info("Extracting object detection data...")
        for prediction, im_shape_path_tuple in self.detector.detector_output:
            output_dict = utils.create_detector_output_dict(
                prediction, im_shape_path_tuple
            )
            for i in output_dict["mask_index"]:
                mask = prediction[0]["masks"][i, 0]  # tensor mask
                mask = self.threshold_mask(mask)  # numpy mask
                mask = cv2.resize(mask, dsize=output_dict["original_image_shape"][::-1])
                output_dict["masks"].append(mask)
            output_stem = Path(output_dict["image_path"]).stem
            # Convert all lists to numpy arrays
            output_dict = {k: np.array(v) for (k, v) in output_dict.items()}
            output_path = Path(self.output_dir, output_stem).with_suffix(".npz")
            logging.info(f"Saving data to {output_path}")
            np.savez_compressed(output_path, **output_dict)

    def threshold_mask(self, mask):
        mask[mask > self.mask_threshold] = 1
        mask[mask <= self.mask_threshold] = 0
        return np.squeeze(mask.numpy().astype(np.uint8))
