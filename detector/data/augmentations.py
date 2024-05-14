import albumentations as A
from albumentations.pytorch.transforms  import ToTensorV2

def get_transform(train):
    if train:
        train_transform = [
            #A.ToFloat(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.RandomGamma(p=0.7),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                    A.HueSaturationValue(p=0.7),
                ],
                p=0.7
            ),
             A.OneOf(
                [
                    A.Sharpen(p=0.3),
                    A.Blur(blur_limit=3, p=0.7)
                ],
                p=0.7,
            ),
            ToTensorV2(),
        ]
    else:
        train_transform = [
            ToTensorV2()
        ]
    return A.Compose(train_transform, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

