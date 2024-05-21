import numpy as np
import pytest
import torch
from detector.legacy import detector_utils as utils
from detector.data.coord_generator import ChimpXtalCoordGenerator


@pytest.fixture
def xtal_coord_generator(echo_detector):
    return ChimpXtalCoordGenerator(echo_detector)


def test_calculate_scale_factors():
    orig_im_shape = (2704, 3376)
    new_im_shape = (1352, 1688)
    scale_y, scale_x = utils.calculate_scale_factors(orig_im_shape, new_im_shape)
    assert scale_y == 2
    assert scale_x == 2
    orig_im_shape = (3000, 1500)
    new_im_shape = (1000, 3000)
    scale_y, scale_x = utils.calculate_scale_factors(orig_im_shape, new_im_shape)
    assert scale_y == 3
    assert scale_x == 0.5


@pytest.mark.parametrize(
    "obj_indices, scale_factors, answer",
    [([25, 25], (2, 3), [50, 75]), ([], (2, 3), []), ([25, 25], (0.5, 0.5), [12, 12])],
)
def test_scale_indices(obj_indices, scale_factors, answer):
    result = utils.scale_indices(obj_indices, scale_factors)
    assert (result == np.array(answer)).all()


def test_threshold_mask(xtal_coord_generator):
    pred, _ = next(xtal_coord_generator.detector.detector_output)
    first_mask = pred[0]["masks"][0]
    assert isinstance(first_mask, torch.Tensor)
    assert len(torch.unique(first_mask)) > 2
    thresh_mask = utils.threshold_mask(first_mask)
    assert isinstance(thresh_mask, np.ndarray)
    assert len(np.unique(thresh_mask)) == 2
