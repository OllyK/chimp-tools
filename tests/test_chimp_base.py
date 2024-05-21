import pytest
from pathlib import Path

import tests.conftest as cfg
from base.chimp_base import ChimpBase
from base.chimp_errors import InputError

MODEL_FILE_NAME = "2022-12-07_CHiMP_Mask_R_CNN_XChem_50eph_VMXi_finetune_DICT_NZ.pytorch"

@pytest.fixture()
def model_path(cwd):
    return cwd.parent/f"classifier/model/{MODEL_FILE_NAME}"

@pytest.fixture()
def chimp_base(model_path, image_folder_real_ims):
    
    class DummyBase(ChimpBase):
        def load_model(self):
            pass
    
    return DummyBase(model_path, list(Path(image_folder_real_ims).glob("*")))

class TestInputParameters():

    def test_good_model_path_good_image_dir_path(self, chimp_base, model_path, image_folder_real_ims):
        assert chimp_base.model_path == model_path
        assert chimp_base.image_list == list(Path(image_folder_real_ims).glob("*"))

    def test_good_model_path_image_dir_path_is_image(self, model_path, image_folder_real_ims):
        with pytest.raises(InputError):
            base = ChimpBase(model_path, image_folder_real_ims/cfg.TEST_IM_FILENAME)

    def test_model_path_is_dir_good_image_dir_path(self, model_path, image_folder_real_ims):
        with pytest.raises(InputError):
            base = ChimpBase(model_path.parent, image_folder_real_ims)

    def test_good_model_path_empty_image_dir(self, model_path, empty_dir):
        with pytest.raises(InputError):
            chimp = ChimpBase(model_path, empty_dir)
