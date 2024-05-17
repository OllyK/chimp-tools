import logging
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from detector.data.augmentations import get_transform
from detector.data.datasets import ZooniverseXtalDropDataset
from utilities import base_data_utils as utils

import utilities.config as cfg


def get_train_val_dataloaders(
    data_dir, settings: SimpleNamespace, fix_seed: bool = False
):
    """
    Get training and validation data loaders.

    Args:
        data_dir (str): Directory path to the data.
        settings (SimpleNamespace): Settings object containing configuration parameters.
        fix_seed (bool, optional): Whether to fix the random seed. Defaults to False.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    training_set_prop = settings.training_set_proportion
    batch_size = utils.get_batch_size(settings)
    labels = settings.class_names

    # Use our dataset and defined transformations
    full_training_dset = ZooniverseXtalDropDataset(
        data_dir, labels, get_transform(train=True)
    )
    full_validation_dset = ZooniverseXtalDropDataset(
        data_dir, labels, get_transform(train=False)
    )

    # Split the dataset into train and validation sets
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

    # Define training and validation data loaders
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        collate_fn=utils.mask_r_cnn_collate_fn,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        collate_fn=utils.mask_r_cnn_collate_fn,
    )

    return training_dataloader, validation_dataloader
