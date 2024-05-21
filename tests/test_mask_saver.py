import pytest
import torch
import numpy as np
from skimage import io

import tests.conftest as cfg
from detector.model.operations.detector_predictor import ChimpDetectorPredictor
from detector.data.mask_saver import ChimpXtalMaskSaver

@pytest.fixture
def mask_saver(echo_detector, empty_dir):
    return ChimpXtalMaskSaver(echo_detector, empty_dir)

class TestBasicFunctions:

    def test_mask_saver_has_detector(self, mask_saver):
        assert isinstance(mask_saver.detector, ChimpDetectorPredictor)

    def test_threshold_mask(self, mask_saver):
        pred, _ = next(mask_saver.detector.detector_output)
        first_mask = pred[0]["masks"][0]
        assert isinstance(first_mask, torch.Tensor)
        assert len(torch.unique(first_mask)) > 2
        thresh_mask = mask_saver.threshold_mask(first_mask)
        assert isinstance(thresh_mask, np.ndarray)
        assert len(np.unique(thresh_mask)) == 2
    
class TestMaskOutputToDisk:

    def test_num_output_files(self, mask_saver, empty_dir):
        mask_saver.extract_masks()
        assert len(list(empty_dir.glob("*.npz"))) == cfg.NUM_REAL_IMS

    def test_saved_file_format(self, mask_saver, empty_dir):
        mask_saver.extract_masks()
        npz_file = list(empty_dir.glob("*.npz"))[0]
        loaded = np.load(npz_file)
        assert "image_path" in loaded
        assert "masks" in loaded

    def test_saved_mask_shape(self, mask_saver, empty_dir):
        mask_saver.extract_masks()
        npz_file = list(empty_dir.glob("*.npz"))[0]
        loaded = np.load(npz_file)
        img = io.imread(str(loaded["image_path"]))
        assert img[:, :, 0].shape == loaded["masks"][0].shape
