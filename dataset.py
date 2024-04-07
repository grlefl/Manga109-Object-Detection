import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_dict, transform=None):
        self.img_dict = img_dict
        self.transform = transform

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        img_path = self.img_dict[idx]['img_path']
        annotations = self.img_dict[idx]['page_annotation']

        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for label in annotations:
            xmin = label['@xmin']
            ymin = label['@ymin']
            xmax = label['@xmax']
            ymax = label['@ymax']

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)  # Assuming the category is the label itself
            areas.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)  # Assuming all annotations are single instances, not crowds

        # Convert lists to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transform:
            image, my_annotation = self.transform(image)

        return image, my_annotation
