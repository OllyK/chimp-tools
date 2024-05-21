import albumentations as A
import numpy as np
import pytest
from pathlib import Path
import torch
from albumentations.pytorch.transforms import ToTensorV2

import tests.conftest as cfg
from detector.legacy.detector_dataset import ChimpDetectorDataset


@pytest.fixture
def transforms():
    return A.Compose([A.Resize(p=1, height=1352, width=1688),
                        A.CLAHE(p=1, tile_grid_size=(12, 12)),
                        ToTensorV2()])

@pytest.fixture
def dataset_no_transforms(image_folder_real_ims):
    return ChimpDetectorDataset(list(Path(image_folder_real_ims).glob("*")))

@pytest.fixture
def dataset_with_transforms(image_folder_real_ims, transforms):
    return ChimpDetectorDataset(list(Path(image_folder_real_ims).glob("*")), transforms=transforms)


class TestBasicFunctions:

    def test_image_list_length(self, dataset_no_transforms):
        assert len(dataset_no_transforms) == cfg.NUM_REAL_IMS 

    def test_image_returned_no_transform(self, dataset_no_transforms):
        assert isinstance(dataset_no_transforms[0][0], np.ndarray)

    def test_original_image_size_no_transform(self, dataset_no_transforms):
        assert isinstance(dataset_no_transforms[0][1], tuple)
 

class TestTransformedData:

    def test_type_returned_after_transform(self, dataset_with_transforms):
        assert isinstance(dataset_with_transforms[0][0], torch.Tensor)

    def test_image_size_after_transform(self, dataset_with_transforms):
        assert dataset_with_transforms[0][0].shape == torch.Size([3, 1352, 1688])

    def test_original_image_size_dimensions_after_transform(self, dataset_with_transforms):
        assert len(dataset_with_transforms[0][1]) == 2 # (shape, Path)
        assert len(dataset_with_transforms[0][1][0]) == 2 # (height, width)