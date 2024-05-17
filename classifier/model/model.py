import logging
from typing import Union
from timm import create_model
import torch

import torch.nn as nn


def create_model_on_device(
    device: str,
    model_struc_dict: dict,
    pretrained=True,
    distributed=False,
) -> torch.nn.Module:
    """
    Create a model on the specified device.

    Args:
        device (str): The device to create the model on (e.g., "cuda", "cpu").
        model_struc_dict (dict): A dictionary containing the model structure information.
        pretrained (bool): Whether to load pretrained weights for the model.
        distributed (bool): Whether to distribute training across multiple GPUs.

    Returns:
        torch.nn.Module: The created model.
    """
    model_type = model_struc_dict["type"]
    num_classes = model_struc_dict["num_classes"]
    model = create_model(
        model_type, pretrained=pretrained, num_classes=num_classes, global_pool="avgmax"
    )

    num_in_features = model.get_classifier().in_features

    if "convnext" in model_type.lower():
        # Add a 'better' classification head
        model.head.fc = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features=num_classes, bias=False),
        )
    elif "resnet" in model_type.lower():
        # Mimic fastai resnet50 head
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Dropout(0.25, inplace=False),
            nn.Linear(in_features=num_in_features, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(in_features=512, out_features=num_classes, bias=True),
        )
    logging.info(f"Sending the {model_type} model to device {device}")
    if "cuda" in device:
        dev_count = torch.cuda.device_count()
        if distributed and dev_count > 1:
            logging.info(f"Distributing training across {dev_count} GPUs.")
            model = nn.DataParallel(model)
    return model.to(device)
