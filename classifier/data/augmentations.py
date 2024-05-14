import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import utilities.config as cfg
import cv2


def get_preprocess_augs(img_size: int) -> A.core.composition.Compose:
    """Returns the augmentations required to pad images to the correct square size
    for training a network.

    Args:
        img_size (int): Length of square image edge.

    Returns:
        A.core.composition.Compose: An augmentation to resize the image if needed.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
        ]
    )


def get_train_augs(img_size: int) -> A.core.composition.Compose:
    """Returns the augmentations used for training a network.

    Returns:
        A.core.composition.Compose: Augmentations for training.
    """
    return A.Compose(
        [
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                p=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0
            ),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.4),
                    A.Blur(blur_limit=3, p=0.4),
                ],
                p=0.3,
            ),
            A.GaussNoise(p=0.2),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.04, p=0.2
                    ),
                    A.GridDistortion(p=0.2),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.2),
                ],
                p=0.4,
            ),
            # A.RandomSizedCrop(
            #     min_max_height=(img_size // 2, img_size),
            #     height=img_size,
            #     width=img_size,
            #     p=0.2,
            # ),
            A.CLAHE(p=0.2),
            A.OneOf([A.RandomBrightnessContrast(p=0.5), A.RandomGamma(p=0.5)], p=0.4),
        ]
    )


def get_postprocess_augs() -> A.core.composition.Compose:
    """Returns the final augmentations applied to the images.

    Returns:
        A.core.composition.Compose: Postprocessing augmentations.
    """
    return A.Compose(
        [A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD), ToTensorV2()]
    )
