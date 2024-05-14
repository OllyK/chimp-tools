from typing import Union

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import logging


def create_mask_r_cnn_on_device(device: Union[int, str], num_classes: int, pretrained: bool = False):
    # load an instance segmentation model pre-trained on COCO
    # TODO: Try v2 of the model
    if pretrained:
        weights = "DEFAULT"
    else:
        weights = None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    logging.info(f"Sending the Mask R CNN model to device {device}")
    return model.to(torch.device(device))
