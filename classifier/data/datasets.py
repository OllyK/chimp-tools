import classifier.data.augmentations as augs
import pandas as pd
from skimage import io
from torch.utils.data import Dataset as BaseDataset
from typing import Union


class ImageClassifierDataset(BaseDataset):
    """Read images, apply augmentations and preprocessing transformations.

    Args:
        input_data (pd.DataFrame): A Pandas dataframe with columns for image path and class label
        preprocessing (albumentations.Compose): data pre-processing
            (e.g. padding, resizing)
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, contrast adjustments)
        postprocessing (albumentations.Compose): data post-processing
            (e.g. Convert to Tensor)
    """

    def __init__(
        self,
        input_data,
        class_to_idx,
        preprocessing=None,
        augmentation=None,
        postprocessing=None,
        is_marco_data=False,
    ):

        self.input_data = input_data
        self.class_to_idx = class_to_idx
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.class_idx = 3 if is_marco_data else 1
        # Create a list with all output labels to enable balancing classes
        self.label_list = (
            self.input_data.iloc[:, self.class_idx].apply(lambda x: self.class_to_idx[x]).tolist()
        )

    def __getitem__(self, i):

        # read data
        image = io.imread(self.input_data.iloc[i, 0])
        idx = self.class_to_idx[self.input_data.iloc[i, self.class_idx]]

        # apply pre-processing
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]

        # apply augmentations
        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        # apply post-processing
        if self.postprocessing:
            image = self.postprocessing(image=image)["image"]

        return image, idx

    def __len__(self):
        return len(self.input_data.index)


class ImageClassifierPredictionDataset(BaseDataset):
    """Read images, apply post and pre-processing transformations.

    Args:
        img_list (list): A list of image filepaths
        preprocessing (albumentations.Compose): data pre-processing
            (e.g. padding, resizing)
        postprocessing (albumentations.Compose): data post-processing
            (e.g. Convert to Tensor)
    """

    def __init__(
        self,
        img_list,
        preprocessing=None,
        postprocessing=None,
    ):
        self.img_list = img_list
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):

        filename = str(self.img_list[i])
        # read data
        image = io.imread(filename)

        # apply pre-processing
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]

        # apply post-processing
        if self.postprocessing:
            image = self.postprocessing(image=image)["image"]
        
        sample = {"image": image, "filename": filename}
        return sample


def get_training_dataset(
    data: pd.DataFrame, img_size: int, class_to_idx: dict, is_marco_data: bool = False,
) -> ImageClassifierDataset:

    return ImageClassifierDataset(
        data,
        class_to_idx,
        preprocessing=augs.get_preprocess_augs(img_size),
        augmentation=augs.get_train_augs(img_size),
        postprocessing=augs.get_postprocess_augs(),
        is_marco_data=is_marco_data
    )


def get_validation_dataset(
    data: pd.DataFrame, img_size: int, class_to_idx: dict, is_marco_data: bool = False,
) -> ImageClassifierDataset:

    return ImageClassifierDataset(
        data,
        class_to_idx,
        preprocessing=augs.get_preprocess_augs(img_size),
        postprocessing=augs.get_postprocess_augs(),
        is_marco_data=is_marco_data
    )


def get_prediction_dataset(
    img_list: list,
    img_size: int,
) -> ImageClassifierPredictionDataset:

    return ImageClassifierPredictionDataset(
        img_list,
        preprocessing=augs.get_preprocess_augs(img_size),
        postprocessing=augs.get_postprocess_augs(),
    )
