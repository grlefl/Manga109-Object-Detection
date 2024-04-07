import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, patches
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_dict, transform=None):
        self.img_dict = img_dict
        self.transform = transform
        self.label_mapping = {'face': 1, 'text': 2, 'frame': 3, 'body': 4}

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        img_path = self.img_dict[idx]['img_path']
        page_annotation = self.img_dict[idx]['page_annotation']

        image = Image.open(img_path).convert("RGB")

        bboxes = []
        labels = []
        areas = []
        iscrowd = []

        valid_categories = ['face', 'text', 'frame', 'body']
        for category in valid_categories:
            if category in page_annotation:
                for box in page_annotation[category]:
                    xmin = box['@xmin']
                    ymin = box['@ymin']
                    xmax = box['@xmax']
                    ymax = box['@ymax']

                    bboxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.label_mapping[category])  # Assuming the category is the label itself
                    areas.append((xmax - xmin) * (ymax - ymin))
                    iscrowd.append(0)  # Assuming all annotations are single instances, not crowds

        # Convert lists to tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        my_annotation = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transform:
            image_np = np.array(image)
            image, my_annotation = self.transform(image=image_np, bboxes=bboxes)

        return image, my_annotation


def visualize_image_with_boxes(image, target):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    boxes = target["boxes"]
    labels = target["labels"]

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'Label: {label}', color='r')

    plt.show()
