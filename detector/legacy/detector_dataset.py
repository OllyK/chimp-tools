from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2


class ChimpDetectorDataset(Dataset):
    """
    A custom dataset class for Chimp Detector.

    Args:
        image_list (list): A list of image paths.
        transforms (callable, optional): A function/transform that takes in an image and returns a transformed version.
                                         Default is None.

    Attributes:
        transforms (callable): A function/transform that takes in an image and returns a transformed version.
        imgs (list): A list of image paths.

    Methods:
        __getitem__(self, idx): Retrieves the image and its corresponding information at the given index.
        __len__(self): Returns the total number of images in the dataset.
    """

    def __init__(self, image_list, transforms=None):
        self.transforms = transforms
        self.imgs = image_list

    def __getitem__(self, idx):
        img_path = Path(self.imgs[idx])
        # load in the image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img_shape = img.shape[:-1]

        if self.transforms is not None:
            transformed = self.transforms(image=img)
            img = transformed["image"]
            i_range = img.max() - img.min()
            img = (img - img.min()) / float(i_range)

        return img, (original_img_shape, img_path)

    def __len__(self):
        return len(self.imgs)
