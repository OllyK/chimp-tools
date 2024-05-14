import logging
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import utilities.base_data_utils as utils
import utilities.config as cfg
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from classifier.data.datasets import (
    get_training_dataset,
    get_validation_dataset,
    get_prediction_dataset,
)

def get_train_val_dataloaders(
    training_data: pd.DataFrame,
    settings: SimpleNamespace,
    imbalanced: bool = False,
    fix_seed: bool = False,
    validation_data: Union[None, pd.DataFrame] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Returns training and validation dataloaders with indices split at random
    according to the percentage split specified in settings.

    Args:
        training_data (pd.DataFrame): A Pandas dataframe with columns for image path and class label
        settings (SimpleNamespace): Settings object

    Returns:
        Tuple[DataLoader, DataLoader]: training and validation dataloaders
    """

    training_set_prop = settings.training_set_proportion
    batch_size = utils.get_batch_size(settings)
    img_size = settings.image_size
    idx_to_class = {i: j for i, j in enumerate(settings.class_names)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    if validation_data is None:
        full_training_dset = get_training_dataset(training_data, img_size, class_to_idx)
        full_validation_dset = get_validation_dataset(training_data, img_size, class_to_idx)
        # split the dataset in train and test set
        dset_length = len(full_training_dset)
        logging.info(f"Full dataset has {dset_length} members.")
        # Set predictable split for testing
        if fix_seed:
            seed_val = settings.random_seed
            logging.info(f"Seeding rng with value {seed_val} from settings.")
            rand_generator = torch.Generator()
            rand_generator.manual_seed(seed_val)
        else:
            rand_generator = None
        indices = torch.randperm(dset_length, generator=rand_generator).tolist()
        train_idx, validate_idx = np.split(indices, [int(dset_length * training_set_prop)])
        logging.info(
            f"Splitting into training dataset: {len(train_idx)} members, validation dataset: {len(validate_idx)} members."
        )
        training_dataset = Subset(full_training_dset, train_idx)
        validation_dataset = Subset(full_validation_dset, validate_idx)
    else:
        training_dataset = get_training_dataset(training_data, img_size, class_to_idx, is_marco_data=True)
        validation_dataset = get_validation_dataset(validation_data, img_size, class_to_idx, is_marco_data=True)

    if imbalanced:  # Oversample under-represented classes
        if validation_data is None:
            label_list = np.array(training_dataset.dataset.label_list)[train_idx]
        else:
            label_list = np.array(training_dataset.label_list)
        training_label_count = np.unique(label_list, return_counts=True)[1]
        weight = 1.0 / torch.tensor(training_label_count, dtype=torch.float)
        class_weights = list(zip(settings.class_names, weight.tolist()))
        logging.info(
            f"Imbalanced data is being sampled with the following weights: {class_weights}."
        )
        sample_weight = weight[label_list]
        sampler = WeightedRandomSampler(
            weights=sample_weight, num_samples=len(sample_weight), replacement=True
        )
    else:
        sampler = None

    training_dataloader = DataLoader(
        training_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader


def get_prediction_dataloader(
    img_list: list, img_size: int = 384, batch_size: int = 24
):
    pred_dataset = get_prediction_dataset(img_list, img_size=img_size)
    return DataLoader(
        pred_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS
    )
