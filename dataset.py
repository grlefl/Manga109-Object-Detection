import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_df, padding=550, transform=None):
        self.img_df = img_df
        self.max_value = padding
        self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df.iloc[idx]['img_path']
        boxes = self.img_df.iloc[idx]['bboxes']         # [[box#1], [box#2], [box#3]]
        labels = self.img_df.iloc[idx]['labels']        # [label#1, label#2, label#3]

        image = cv2.imread(img_path)                            # load the image as numpy array (BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # convert BGR image to RGB format
        image = image.astype(np.float32) / 255.                 # normalize image for pixel range [0, 1]

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)                       # bounding box to tensor
        # # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])          # area of the bounding boxes
        # # iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)               # no crowd instances
        # labels = torch.as_tensor(labels, dtype=torch.int64)                         # labels to tensor

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        # image_id = torch.tensor([idx])
        # target["image_id"] = image_id

        # apply the image transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=target['boxes'], labels=labels)
            image = transformed['image']
            target['boxes'] = torch.Tensor(transformed['bboxes'])   # format to tensor
            target['images'] = torch.Tensor(transformed['labels'])  # format to tensor

        return image, target
    

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
