import cv2
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
        bboxes = self.img_df.iloc[idx]['bboxes']        # [[box#1], [box#2], [box#3]]
        labels = self.img_df.iloc[idx]['labels']        # [label#1, label#2, label#3]

        image = cv2.imread(img_path)                        # load the image as numpy array (BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # convert BGR image to RGB format
        # image = image.astype(np.float32) / 255.             # normalize image for pixel range [0, 1]

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']    # including image ToTensor
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        # This portion of code converts 'bboxes' and 'labels' to tensors.
        # (I tried to include all tensor transforms in a custom transform, but the albumentations
        # library isn't very friendly to work with and causing errors.)

        # for tensor uniformity, 'bboxes' and 'labels' are padded to reach the 'padding' value
        num_boxes = len(bboxes)
        pad_size = self.max_value - num_boxes
        if pad_size > 0:
            padded_bboxes = bboxes + [[0, 0, 0, 0]] * pad_size
            padded_labels = labels + [0] * pad_size
        else:
            padded_bboxes = bboxes  # [:self.max_value]
            padded_labels = labels  # [:self.max_value]

        # convert padded bounding boxes and labels to tensors
        bboxes_tensor = torch.tensor(padded_bboxes, dtype=torch.float32)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.int64)

        return image, bboxes_tensor, labels_tensor


# class ToTensor(object):
#     def __init__(self, max_value=10):
#         self.max_value = max_value
#
#     def __call__(self, image, bboxes, labels):
#         # convert image to tensor
#         image_tensor = F.to_tensor(image)
#
#         # pad bounding boxes and labels
#         num_boxes = len(bboxes)
#         pad_size = self.max_value - num_boxes
#         if pad_size > 0:
#             padded_bboxes = bboxes + [[0, 0, 0, 0]] * pad_size
#             padded_labels = labels + [0] * pad_size
#         else:
#             padded_bboxes = bboxes[:self.max_value]
#             padded_labels = labels[:self.max_value]
#
#         # convert padded bounding boxes and labels to tensors
#         bboxes_tensor = torch.tensor(padded_bboxes, dtype=torch.float32)
#         labels_tensor = torch.tensor(padded_labels, dtype=torch.int64)
#
#         return image_tensor, bboxes_tensor, labels_tensor
