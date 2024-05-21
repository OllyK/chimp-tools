from torchvision.models.detection.mask_rcnn import MaskRCNN
import types
from pathlib import Path

import tests.conftest as cfg

class TestFixtures():

    def test_intial_image_list_length(self, image_folder_real_ims):
        assert len(list(image_folder_real_ims.glob("*.jpg"))) == cfg.NUM_REAL_IMS

    def test_model_file_exists(self, echo_detector_model_path):
        assert echo_detector_model_path.is_file()

class TestModelLoading():

    def test_chimp_load_model_type(self, echo_detector):
        assert isinstance(echo_detector.model, MaskRCNN)

class TestDataLoading():
    
    def test_chimp_load_data(self, echo_detector):
        assert len(echo_detector.dataset) == cfg.NUM_REAL_IMS

class TestPrediction():
    
    def test_detector_predict_single_image(self, echo_detector):
        image = echo_detector.dataset[0][0]
        prediction = echo_detector.predict_single_image(image)
        assert isinstance(prediction, list)
        assert len(prediction) == 1
        assert "masks" in prediction[0]

    def test_detector_predict_output_generator(self, echo_detector):
        assert isinstance(echo_detector.detector_output, types.GeneratorType)
        prediction = next(echo_detector.detector_output)
        assert isinstance(prediction, tuple)
        assert len(prediction) == 2
        assert "masks" in prediction[0][0]
