import logging
import os

import cv2
import numpy as np
import torch


class ZooniverseXtalDropDataset(torch.utils.data.Dataset):
    def __init__(self, root, labels, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        self.class_key = {v: k + 1 for k, v in enumerate(labels)}
        logging.info(f"Class key: {self.class_key}")
        
    def __getitem__(self, idx):
        empty_flag= False
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        # load in the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # load in the corresponding mask and class label list
        mask_file = np.load(mask_path)
        masks = list(mask_file["masks"].astype(int))
#         masks = np.array(masks).astype(int)
        class_labels = list(mask_file["class_labels"])
        
        # get bounding box coordinates for each mask
        num_objs = len(masks)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        if num_objs == 0:
            empty_flag = True
            area = [0]
            num_objs = 1
        else:
            boxes = np.array(boxes, dtype=np.float32)
     
        if self.transforms is not None and not empty_flag:
            transformed = self.transforms(image=img, masks=masks, bboxes=boxes, class_labels=class_labels)
            img = transformed['image']
            img = self.normalise_image(img)
            masks = transformed['masks']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            labels = [self.class_key.get(item) for item in class_labels]
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if empty_flag:
            labels = torch.as_tensor(np.array([0], dtype=int), dtype=torch.int64)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            img = self.normalise_image(img)
            boxes = np.array([[0, 1, 2, 3]], dtype=np.float32)
            masks = [torch.zeros(( img.shape[1], img.shape[2]), dtype=torch.uint8)]

        boxes = torch.as_tensor(boxes, dtype=torch.float64)    
        if not empty_flag:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = torch.stack(masks).byte()
        target["image_id"] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        return img, target
    
    def normalise_image(self, img):
        i_range = img.max() - img.min()
        return (img - img.min())/i_range

    def __len__(self):
        return len(self.imgs)

    