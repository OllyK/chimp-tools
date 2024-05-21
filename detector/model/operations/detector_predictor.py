import logging

import albumentations as A
import torch
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import detector.legacy.detector_config as cfg
from base.chimp_base import ChimpBase
from detector.legacy.detector_dataset import ChimpDetectorDataset


class ChimpDetectorPredictor(ChimpBase):

    def __init__(self, model_path, image_list, num_classes, transforms=True):
        super().__init__(model_path, image_list)
        self.num_classes = num_classes
        self.model = self.load_model()
        self.transforms = self.get_transforms() if transforms else None
        self.dataset = self.setup_dataset()
        self.detector_output = self.generate_all_predictions()

    def load_model(self):
        logging.info(f"Loading model from {self.model_path}")
        model = self.get_instance_segmentation_model(self.num_classes)
        model.load_state_dict(torch.load(str(self.model_path), map_location=torch.device('cpu')))
        model.eval()
        return model

    def get_instance_segmentation_model(self, num_classes):
        # load an instance segmentation model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = cfg.MODEL_HIDDEN_LAYER_SIZE
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
        return model

    def get_transforms(self):
        return A.Compose([A.Resize(p=1, height=cfg.IM_RESIZE_HEIGHT, width=cfg.IM_RESIZE_WIDTH),
                          A.CLAHE(p=1, tile_grid_size=cfg.CLAHE_GRID_SIZE),
                          ToTensorV2()])

    def setup_dataset(self):
            return ChimpDetectorDataset(self.image_list, transforms=self.transforms)

    def predict_single_image(self, image):
        with torch.no_grad():
            prediction = self.model([image])
        return prediction

    def generate_all_predictions(self):
        for image, im_shape_path_tuple in self.dataset:
            yield self.predict_single_image(image), im_shape_path_tuple
